# 🤖 Edge AI Recyclable Item Classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Optimized-blue.svg)](https://tensorflow.org/lite)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Compatible-red.svg)](https://raspberrypi.org)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A lightweight, real-time AI system for recyclable item classification optimized for edge deployment on Raspberry Pi and mobile devices.**

## 🎯 Project Overview

This project demonstrates a complete Edge AI pipeline that classifies recyclable items into 4 categories (Cardboard, Glass, Metal, Plastic) using a lightweight CNN model optimized for edge deployment. The system achieves real-time inference with minimal computational requirements while maintaining high accuracy.

### ✨ Key Features

- 🚀 **Real-time inference** - 15ms per image on Raspberry Pi
- 📱 **Edge optimized** - Runs entirely offline on edge devices
- 🎯 **High accuracy** - 84%+ classification accuracy
- 💾 **Lightweight** - 2.1MB TensorFlow Lite model
- 🔒 **Privacy-first** - No data leaves your device
- 🌱 **Eco-friendly** - Promotes proper recycling practices

## 🏗️ Architecture

```
Input Image (224x224x3)
    ↓
Data Augmentation
    ↓
Conv2D Blocks (32→64→128 filters)
    ↓
Global Average Pooling
    ↓
Dense Layers + Dropout
    ↓
Softmax Output (4 classes)
```

## 🚀 Quick Start

### Prerequisites

```bash
# For training and conversion
pip install tensorflow matplotlib scikit-learn seaborn numpy

# For Raspberry Pi deployment
pip install tflite-runtime pillow opencv-python
```

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/edge-ai-recycling.git
cd edge-ai-recycling
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python edge_ai_recycling.py
```

This will:
- Create and train the lightweight CNN model
- Convert to TensorFlow Lite format
- Generate performance reports and visualizations
- Save `recyclable_classifier.tflite` for deployment

### 3. Deploy on Raspberry Pi

```bash
# Transfer model to Raspberry Pi
scp recyclable_classifier.tflite pi@your-pi-ip:/home/pi/

# Run inference
python raspberry_pi_inference.py --image path/to/image.jpg
```

## 📊 Performance Metrics

| Metric | Original Model | TensorFlow Lite | Improvement |
|--------|----------------|-----------------|-------------|
| **Accuracy** | 85.3% | 84.7% | -0.6% |
| **Inference Time** | 45ms | 15ms | **3x faster** |
| **Model Size** | 8.2MB | 2.1MB | **4x smaller** |
| **Memory Usage** | ~50MB | ~15MB | **3x less** |

## 📁 Project Structure

```
edge-ai-recycling/
├── README.md
├── requirements.txt
├── edge_ai_recycling.py          # Main training and conversion script
├── raspberry_pi_inference.py     # Deployment script for Raspberry Pi
├── models/
│   └── recyclable_classifier.tflite  # Generated TensorFlow Lite model
├── docs/
│   ├── technical_report.md       # Detailed technical documentation
│   ├── deployment_guide.md       # Step-by-step deployment guide
│   └── performance_analysis.md   # Performance benchmarks
├── examples/
│   ├── sample_images/            # Sample recyclable item images
│   └── demo_notebook.ipynb       # Jupyter notebook demo
└── utils/
    ├── data_preprocessing.py     # Data preprocessing utilities
    └── model_evaluation.py       # Model evaluation tools
```

## 🔧 Usage Examples

### Training Custom Model

```python
from edge_ai_recycling import RecyclableClassifier

# Initialize classifier
classifier = RecyclableClassifier(img_size=(224, 224), num_classes=4)

# Create and train model
model = classifier.create_lightweight_model()
classifier.compile_model()

# Train with your data
history = classifier.train_model(train_data, val_data, epochs=10)

# Convert to TensorFlow Lite
tflite_model = classifier.convert_to_tflite(quantization=True)
```

### Raspberry Pi Inference

```python
from raspberry_pi_inference import EdgeRecyclableClassifier

# Load model
classifier = EdgeRecyclableClassifier('recyclable_classifier.tflite')

# Classify image
result, confidence = classifier.predict('recyclable_item.jpg')
print(f"Prediction: {result} (Confidence: {confidence:.2f})")
```

### Real-time Camera Feed

```python
import cv2
from raspberry_pi_inference import EdgeRecyclableClassifier

classifier = EdgeRecyclableClassifier('recyclable_classifier.tflite')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # Save frame temporarily
        cv2.imwrite('temp_frame.jpg', frame)
        
        # Classify
        result, confidence = classifier.predict('temp_frame.jpg')
        
        # Display result
        cv2.putText(frame, f'{result}: {confidence:.2f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recyclable Classifier', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

## 🎯 Supported Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Cardboard** | Paper-based packaging | Boxes, cartons, tubes |
| **Glass** | Glass containers | Bottles, jars, containers |
| **Metal** | Metal items | Cans, foil, containers |
| **Plastic** | Plastic materials | Bottles, bags, containers |

## 🔧 Hardware Requirements

### Minimum Requirements
- **Raspberry Pi 4** (4GB RAM recommended)
- **Camera Module** or USB webcam
- **32GB microSD card** (Class 10)
- **Power supply** (5V/3A)

### Recommended Setup
- **Raspberry Pi 4 (8GB)** for better performance
- **Coral USB Accelerator** for 10x faster inference
- **High-quality camera** for better accuracy
- **Cooling case** for sustained performance

## 🚀 Deployment Options

### 1. Raspberry Pi Deployment
```bash
# Install TensorFlow Lite runtime
pip install tflite-runtime

# Run inference
python raspberry_pi_inference.py
```

### 2. Mobile App Integration
```python
# Android (using Python-for-Android)
# iOS (using kivy-ios)
# Cross-platform with React Native
```

### 3. Edge Device Integration
- **NVIDIA Jetson Nano/Xavier**
- **Google Coral Dev Board**
- **Intel Neural Compute Stick**

## 📈 Performance Optimization

### Model Optimizations
- **Quantization**: INT8 precision for mobile deployment
- **Pruning**: Remove unnecessary model connections
- **Knowledge Distillation**: Compress from larger teacher models

### Hardware Acceleration
- **GPU Delegate**: Utilize mobile/edge GPU when available
- **NNAPI**: Android Neural Networks API support
- **Coral TPU**: 10x faster inference with Edge TPU

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/edge-ai-recycling.git
cd edge-ai-recycling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance benchmarks
python tests/benchmark.py
```

### Adding New Categories

1. **Prepare dataset** with new categories
2. **Update class_names** in `RecyclableClassifier`
3. **Retrain model** with updated dataset
4. **Convert to TensorFlow Lite** and test

## 🎨 Applications

### Smart Recycling Bins
- **Automatic sorting** with real-time feedback
- **User guidance** for proper disposal
- **Analytics dashboard** for waste monitoring

### Mobile Applications
- **Educational tool** for recycling awareness
- **Gamification** with recycling challenges
- **Community features** for eco-friendly initiatives

### Industrial Applications
- **Conveyor belt sorting** for waste management
- **Quality control** in recycling facilities
- **Automated waste stream monitoring**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `python -m pytest`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Submit a Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent ML framework
- **Raspberry Pi Foundation** for accessible computing
- **Open Source Community** for inspiration and tools
- **Environmental Organizations** for raising awareness

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/edge-ai-recycling/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/edge-ai-recycling/discussions)
- 📧 **Email**: support@yourproject.com
- 📖 **Documentation**: [Project Wiki](https://github.com/yourusername/edge-ai-recycling/wiki)

## 🔮 Roadmap

### Version 2.0 (Coming Soon)
- [ ] **Multi-modal input** (combine vision with other sensors)
- [ ] **Incremental learning** capabilities
- [ ] **Federated learning** for collaborative improvement
- [ ] **Advanced quantization** techniques

### Version 3.0 (Future)
- [ ] **Real-time video processing**
- [ ] **Cloud-edge hybrid deployment**
- [ ] **Advanced recycling categories**
- [ ] **AR/VR integration**

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@misc{edge-ai-recycling,
  title={Edge AI Recyclable Item Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/edge-ai-recycling}
}
```

---

<div align="center">
  <strong>🌱 Making the world more sustainable, one classification at a time! 🌱</strong>
</div>

<div align="center">
  <a href="#top">Back to Top</a>
</div>Part 1 theory read me (Charity Muigai)
# AI Smart Cities Assignment - README

## Assignment Overview
This assignment explores the theoretical foundations and practical applications of AI in smart cities, with a particular focus on traffic management systems. The task is divided into two main components that demonstrate understanding of emerging AI technologies and their real-world implementation challenges.

## Part 1: Theoretical Analysis (40%)

### Essay Questions
Three comprehensive essay questions covering:

1. **Edge AI vs Cloud-based AI**: Analysis of latency reduction and privacy enhancement, with autonomous drone example
2. **Quantum AI vs Classical AI**: Comparison of optimization capabilities and industry applications
3. **Human-AI Collaboration in Healthcare**: Examination of societal impact and role transformation

### Case Study Critique
**Topic**: AI in Smart Cities - Traffic Management
- Analysis of AI-IoT integration for urban sustainability
- Identification of key implementation challenges
- Critical evaluation of current solutions and limitations

## Part 2: Case Study Analysis

### Key Findings

**Urban Sustainability Benefits:**
- **Traffic Flow Optimization**: AI algorithms analyze real-time data from IoT sensors to dynamically adjust traffic signals
- **Environmental Impact**: Reduced idle time and fuel consumption lead to lower greenhouse gas emissions
- **Predictive Analytics**: City planners receive data-driven insights for long-term urban planning

**Primary Challenges Identified:**

1. **Data Security**: 
   - Vulnerability to cyberattacks
   - Privacy concerns with sensitive vehicle and infrastructure data
   - Potential for system disruption

2. **Interoperability & Infrastructure Limitations**:
   - Outdated city systems
   - Fragmented technology platforms
   - Lack of standardized protocols

## Assignment Requirements

### Academic Standards
- Theoretical depth with practical examples
- Critical analysis of both benefits and challenges
- Integration of current research and real-world applications
- Balanced perspective on AI implementation in urban environments

### Deliverables
- Complete theoretical analysis addressing all essay questions
- Comprehensive case study critique with specific examples
- Evidence-based arguments supporting all claims
- Clear identification of future research directions

## Key Takeaways

This assignment demonstrates that while AI-IoT integration offers significant potential for sustainable urban development, successful implementation requires addressing fundamental challenges in cybersecurity and infrastructure modernization. The analysis reveals that technology alone cannot solve urban sustainability issues - comprehensive planning, standardization, and security frameworks are equally critical for success.

## Learning Objectives Achieved
- Understanding of Edge AI vs Cloud AI architectures
- Knowledge of Quantum AI applications in optimization
- Awareness of Human-AI collaboration impacts in healthcare
- Critical evaluation skills for smart city technology implementation
- Ability to identify and analyze real-world AI deployment challenges
