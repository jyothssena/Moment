# MODEL SENSITIVITY ANALYSIS REPORT — MOMENT PROJECT


Date: 2026-03-30

Total rows analysed: 4880

Features analysed: gender, age_group, personality, reader_type

Scores analysed: think_R, feel_R


============================================================

## 1. CORRELATION ANALYSIS


Eta-squared measures how much variance in each score is explained by each demographic feature.


| Feature | think_R η² | feel_R η² | think interpretation | feel interpretation |

|---|---|---|---|---|

| gender | 0.0018 | 0.0022 | negligible | negligible |

| age_group | 0.0007 | 0.0009 | negligible | negligible |

| personality | 0.0037 | 0.0036 | negligible | negligible |

| reader_type | 0.0035 | 0.0057 | negligible | negligible |


*p < 0.05 indicates statistically significant relationship.


============================================================

## 2. GROUP MEAN COMPARISON


Mean scores per demographic group. 
Flags groups deviating more than 1 std from overall mean.


### gender


| Group | Count | mean think_R | mean feel_R | think flag | feel flag |

|---|---|---|---|---|---|

| Female | 2313 | 59.4 | 49.3 | ✓ | ✓ |

| Male | 2567 | 57.5 | 47.2 | ✓ | ✓ |

| **Overall** | 4880 | 58.4 | 48.2 | - | - |


### age_group


| Group | Count | mean think_R | mean feel_R | think flag | feel flag |

|---|---|---|---|---|---|

| 18-24 (Gen Z) | 1040 | 59.3 | 48.8 | ✓ | ✓ |

| 25-34 (Millennial) | 2284 | 58.2 | 47.7 | ✓ | ✓ |

| 35-44 (Gen X/Mill) | 1010 | 57.7 | 49.1 | ✓ | ✓ |

| 45+ (Gen X/Boom) | 546 | 58.9 | 47.2 | ✓ | ✓ |

| **Overall** | 4880 | 58.4 | 48.2 | - | - |


### personality


| Group | Count | mean think_R | mean feel_R | think flag | feel flag |

|---|---|---|---|---|---|

| Analytical | 1751 | 56.8 | 46.5 | ✓ | ✓ |

| Emotional | 1485 | 59.8 | 49.5 | ✓ | ✓ |

| Narrative | 640 | 57.9 | 48.1 | ✓ | ✓ |

| Philosophical | 1004 | 59.4 | 49.1 | ✓ | ✓ |

| **Overall** | 4880 | 58.4 | 48.2 | - | - |


### reader_type


| Group | Count | mean think_R | mean feel_R | think flag | feel flag |

|---|---|---|---|---|---|

| ACCIDENTAL | 164 | 56.6 | 48.4 | ✓ | ✓ |

| DELIBERATE | 1350 | 59.7 | 49.1 | ✓ | ✓ |

| HABITUAL | 893 | 56.9 | 48.2 | ✓ | ✓ |

| NEW READER | 1279 | 59.2 | 48.3 | ✓ | ✓ |

| PROJECT | 532 | 56.4 | 43.7 | ✓ | ✓ |

| SOCIAL | 662 | 58.4 | 49.5 | ✓ | ✓ |

| **Overall** | 4880 | 58.4 | 48.2 | - | - |


============================================================

## 3. FEATURE IMPORTANCE RANKING


Features ranked by eta-squared.


Higher eta-squared = more important feature.


### think_R


| Rank | Feature | Eta-squared | Interpretation |

|---|---|---|---|

| 1 | personality | 0.0037 | negligible |

| 2 | reader_type | 0.0035 | negligible |

| 3 | gender | 0.0018 | negligible |

| 4 | age_group | 0.0007 | negligible |



### feel_R


| Rank | Feature | Eta-squared | Interpretation |

|---|---|---|---|

| 1 | reader_type | 0.0057 | negligible |

| 2 | personality | 0.0036 | negligible |

| 3 | gender | 0.0022 | negligible |

| 4 | age_group | 0.0009 | negligible |




============================================================

## SUMMARY


**think_R**

- Most influential feature: `personality` (η²=0.0037, negligible effect)

- Least influential feature: `age_group` (η²=0.0007, negligible effect)


**feel_R**

- Most influential feature: `reader_type` (η²=0.0057, negligible effect)

- Least influential feature: `age_group` (η²=0.0009, negligible effect)


### Interpretation guide


- **Negligible (η² < 0.01)**: feature has almost no influence on scores

- **Small (0.01–0.06)**: feature has minor influence

- **Medium (0.06–0.14)**: feature has moderate influence

- **Large (η² > 0.14)**: feature has strong influence on scores


A large effect means the model's scores vary significantly across groups for that feature — worth investigating whether this reflects genuine reader differences or potential model sensitivity.
