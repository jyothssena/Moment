#!/bin/bash
set -e

PROJECT_ID="${GCP_PROJECT_ID:-moment-486719}"
REGION="${GCP_REGION:-us-central1}"
CLUSTER_NAME="moment-cluster"
REGISTRY_REPO="moment-agents"
GCS_BUCKET="${GCS_BUCKET:-moment-agent-data}"
CICD_SA_NAME="moment-cicd-sa"
WORKLOAD_SA_NAME="moment-workload-sa"
K8S_NAMESPACE="moment"
K8S_SA_NAME="moment-ksa"

echo ""
echo "============================================================"
echo " Moment — Full GCP Infrastructure Setup"
echo "============================================================"
echo " Project: $PROJECT_ID  |  Region: $REGION"
echo " Cluster: $CLUSTER_NAME"
echo "============================================================"
echo ""

echo "Step 1/7: Enabling GCP APIs..."
gcloud services enable \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  secretmanager.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  --project="$PROJECT_ID" --quiet
echo "  Done."

echo ""
echo "Step 2/7: Creating Artifact Registry..."
gcloud artifacts repositories create "$REGISTRY_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Moment Docker images" \
  --project="$PROJECT_ID" 2>/dev/null || echo "  Already exists."

echo ""
echo "Step 3/7: Creating GCS bucket..."
gcloud storage buckets create "gs://$GCS_BUCKET" \
  --location="$REGION" --project="$PROJECT_ID" 2>/dev/null || echo "  Already exists."
echo '{}' | gcloud storage cp - "gs://$GCS_BUCKET/metrics_baseline.json"
echo "  Seeded metrics_baseline.json"

echo ""
echo "Step 4/7: Creating GKE cluster (takes 3-5 minutes)..."
if ! gcloud container clusters describe "$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
# Create Cloud Router + NAT (private nodes need this for internet access)
  gcloud compute routers create moment-router \
    --region="$REGION" --project="$PROJECT_ID" \
    --network=default 2>/dev/null || echo "  Router already exists."

  gcloud compute routers nats create moment-nat \
    --router=moment-router --region="$REGION" --project="$PROJECT_ID" \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges 2>/dev/null || echo "  NAT already exists."

  # Create private cluster (no external IPs on nodes — required by university policy)
  gcloud container clusters create "$CLUSTER_NAME" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --machine-type="e2-standard-4" \
    --num-nodes=1 \
    --min-nodes=1 \
    --max-nodes=3 \
    --enable-autoscaling \
    --workload-pool="${PROJECT_ID}.svc.id.goog" \
    --enable-ip-alias \
    --enable-private-nodes \
    --master-ipv4-cidr=172.16.0.0/28 \
    --no-enable-master-authorized-networks \
    --spot \
    --quiet
else
  echo "  Already exists."
fi

gcloud container clusters get-credentials "$CLUSTER_NAME" \
  --region="$REGION" --project="$PROJECT_ID" --quiet
echo "  kubectl configured."

echo ""
echo "Step 5/7: Setting up Kubernetes namespace + Workload Identity..."
kubectl create namespace "$K8S_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

WORKLOAD_SA_EMAIL="${WORKLOAD_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create "$WORKLOAD_SA_NAME" \
  --display-name="Moment Workload Identity SA" \
  --project="$PROJECT_ID" 2>/dev/null || echo "  SA already exists."

for ROLE in roles/storage.objectAdmin roles/bigquery.dataEditor roles/bigquery.jobUser roles/aiplatform.user; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$WORKLOAD_SA_EMAIL" --role="$ROLE" --quiet
done

kubectl create serviceaccount "$K8S_SA_NAME" \
  --namespace="$K8S_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

gcloud iam service-accounts add-iam-policy-binding "$WORKLOAD_SA_EMAIL" \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${K8S_NAMESPACE}/${K8S_SA_NAME}]" \
  --project="$PROJECT_ID" --quiet

kubectl annotate serviceaccount "$K8S_SA_NAME" --namespace="$K8S_NAMESPACE" \
  iam.gke.io/gcp-service-account="$WORKLOAD_SA_EMAIL" --overwrite
echo "  Workload Identity ready: pods access GCS/BQ without key files."

echo ""
echo "Step 6/7: Creating CI/CD service account..."
CICD_SA_EMAIL="${CICD_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create "$CICD_SA_NAME" \
  --display-name="Moment CI/CD SA" --project="$PROJECT_ID" 2>/dev/null || echo "  Already exists."

for ROLE in roles/artifactregistry.writer roles/container.developer roles/storage.objectAdmin roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$CICD_SA_EMAIL" --role="$ROLE" --quiet
  echo "  Granted: $ROLE"
done

KEY_FILE="./moment-cicd-sa-key.json"
gcloud iam service-accounts keys create "$KEY_FILE" \
  --iam-account="$CICD_SA_EMAIL" --project="$PROJECT_ID"
echo "  Key saved: $KEY_FILE"

echo ""
echo "Step 7/7: GitHub Secrets to add:"
echo ""
echo "  GCP_SA_KEY           → run: cat $KEY_FILE | base64 -w 0"
echo "  GCP_PROJECT_ID       → $PROJECT_ID"
echo "  GCP_REGION           → $REGION"
echo "  GKE_CLUSTER_NAME     → $CLUSTER_NAME"
echo "  GCS_BUCKET           → $GCS_BUCKET"
echo "  GEMINI_API_KEY_MOMENT → your Gemini API key"
echo "  SLACK_WEBHOOK_URL    → your Slack webhook (optional)"
echo ""
echo "============================================================"
echo " Done. Push to main to trigger first deployment."
echo " Security: rm $KEY_FILE  (after adding to GitHub)"
echo "============================================================"
