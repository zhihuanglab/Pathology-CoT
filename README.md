# Pathology-CoT

This is for paper: Pathology-CoT: Learning Visual Chain-of-Thought Agent from Expert Whole Slide Image Diagnosis Behavior

Diagnosing a whole-slide image diagnosis is an interactive, multi-stage process of changing magnification and moving between fields. 
Although recent pathology foundation models are strong, practical agentic systems that decide what field to examine next, adjust magnification, and deliver explainable diagnoses are still lacking.
The blocker is data: scalable, clinically aligned supervision of expert viewing behavior that is tacit and experience‑based, not written in textbooks or online, and therefore absent from LLM training.
We introduce the AI Session Recorder, which works with standard WSI viewers to unobtrusively record routine navigation and convert the viewer logs into standardized behavioral commands (inspect/peek at discrete magnifications) and bounding boxes.
A lightweight human-in-the-loop review turns AI-drafted rationales into the \textbf{Pathology-CoT dataset}, a form of paired ``where to look” and ``why it matters” supervision produced at a roughly six-fold lower labeling time.
Using this behavioral data, we build \textbf{Pathologist-o3}, a two-stage agent that first proposes ROIs and then performs behavior-guided reasoning. 
On gastrointestinal lymph-node metastasis detection it achieved 84.5\% precision, 100.0\% recall, and 75.4\% accuracy, exceeding the state-of-the-art OpenAI o3 model and generalizing across backbones. 
To our knowledge, this constitutes one of the first behavior-grounded agentic systems in pathology.
Turning everyday viewer logs into scalable, expert‑validated supervision, our framework makes agentic pathology practical and establishes a path to human‑aligned, upgradeable clinical AI.