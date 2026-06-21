# 🤖 Agentic AI & n8n Automation Powered Maternal Care Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![AI](https://img.shields.io/badge/AI-Agentic%20AI-orange)
![Automation](https://img.shields.io/badge/Automation-n8n-red)
![Healthcare](https://img.shields.io/badge/Domain-Maternal%20Healthcare-pink)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

## 🩺 Overview

An intelligent maternal healthcare monitoring framework that combines **Agentic AI**, **Machine Learning**, and **n8n Workflow Automation** to provide continuous pregnancy monitoring, predictive risk assessment, automated clinical triage, and personalized healthcare recommendations.

Traditional maternal care relies on periodic clinical visits, which may fail to detect rapidly developing complications such as **Preeclampsia** and **Gestational Diabetes**. This framework bridges that gap through continuous monitoring, intelligent reasoning, and automated intervention workflows.

---

## 🎯 Project Goals

✅ Continuous Maternal Health Monitoring

✅ Early Detection of Pregnancy Complications

✅ Predictive Risk Assessment Using Machine Learning

✅ AI-Powered Clinical Decision Support

✅ Automated Workflow Orchestration with n8n

✅ Context-Aware Alert Generation

✅ Continuous Learning Through Expert Feedback

---

## 🚨 Problems Addressed

### 📊 Data Fragmentation

Patient information is often distributed across multiple systems such as EHRs, wearable devices, and patient-reported logs.

### 🔔 Alert Fatigue

Static threshold-based systems generate excessive false alarms, reducing clinician trust.

### ⏳ Delayed Intervention

Traditional monitoring may miss subtle warning signs that develop between scheduled clinical visits.

### 🧠 Lack of Contextual Intelligence

Conventional monitoring systems analyze numbers but fail to understand symptom context.

---

## ✨ Key Features

### 📈 Predictive Risk Analytics

* XGBoost-based risk prediction model
* Early identification of:

  * Preeclampsia
  * Gestational Hypertension
  * Gestational Diabetes
  * Macrosomia Risk

### 🤖 Agentic AI Clinical Reasoning

* Context-aware risk interpretation
* Explainable clinical summaries
* Intelligent patient triage
* Personalized recommendations

### 🔄 n8n Workflow Automation

* Automated data collection
* Scheduled risk assessment
* Alert orchestration
* Multi-channel notifications

### 📱 Smart Alerting System

* Clinician emergency alerts
* Patient self-care notifications
* Risk-based alert prioritization

### 🔁 Continuous Learning

* Human-in-the-loop feedback
* Alert validation
* Dataset enrichment
* Future model retraining

---

## 🏗️ System Architecture

```text
┌─────────────────────┐
│  Patient Data Sources │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Fusion Pipeline │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ XGBoost Risk Engine │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Agentic AI Layer    │
│ (Gemini)            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ n8n Workflow Engine │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
👨‍⚕️ Clinician   👩‍🍼 Patient
 Alerts         Coaching
```

---

## 🛠️ Technology Stack

| Category            | Technologies           |
| ------------------- | ---------------------- |
| 🤖 AI               | Gemini, Agentic AI     |
| 📈 Machine Learning | XGBoost                |
| 🔄 Automation       | n8n                    |
| 🌐 APIs             | REST APIs, Webhooks    |
| 🗄️ Data Sources    | EHR, IoT Devices, PROs |
| 📩 Notifications    | Email, SMS             |
| ☁️ Integration      | Cloud-Based Services   |

---

## 🔄 Workflow
<img width="1126" height="404" alt="image" src="https://github.com/user-attachments/assets/40f0e35f-3538-4880-aec5-4d6ce7fbd713" />

```mermaid
flowchart TD
    A[Patient Data] --> B[Data Processing]
    B --> C[XGBoost Risk Prediction]
    C --> D[Agentic AI Analysis]
    D --> E[n8n Automation]
    E --> F[Clinical Alerts]
    E --> G[Patient Notifications]
    F --> H[Clinician Feedback]
    H --> I[Continuous Learning]
```

---

## 📊 Expected Outcomes

🎯 Early Detection of Maternal Complications

🎯 Reduced Clinical Alert Fatigue

🎯 Improved Patient Safety

🎯 Faster Emergency Response

🎯 Enhanced Clinical Decision Support

🎯 Continuous System Improvement

---

## 🚀 Future Enhancements

* 📱 Mobile Application
* 🌍 Multi-language Support
* 🏥 FHIR & HL7 Integration
* 🔒 Federated Learning
* ⌚ Real-Time Wearable Integration
* 📊 Advanced Explainable AI Dashboards

---

## 👨‍💻 Author

**Subhash Thippa**
Integrated M.Tech Software Engineering
Vellore Institute of Technology (VIT), Chennai

---

## 📄 License

This project is developed for academic and research purposes and is intended for educational and non-commercial use.
