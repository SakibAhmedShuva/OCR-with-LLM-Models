# OCR with Large Language Models (LLMs)

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/SakibAhmedShuva/OCR-with-LLM-Models?style=social)](https://github.com/SakibAhmedShuva/OCR-with-LLM-Models/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/SakibAhmedShuva/OCR-with-LLM-Models?style=social)](https://github.com/SakibAhmedShuva/OCR-with-LLM-Models/network/members)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## 📝 Overview

This repository explores the capabilities of various open-source and accessible Large Language Models (LLMs) with vision capabilities for performing Optical Character Recognition (OCR). Each notebook demonstrates how to use a specific model to extract text from a sample image (a California Driver's License).

The primary goal is to perform meticulous and verbatim text extraction, capturing all characters, numbers, and symbols exactly as they appear on the document image.

## 🔍 Models Explored

This repository contains Jupyter Notebooks for the following models:

| Model | Notebook | Hugging Face | Try it |
|-------|----------|--------------|--------|
| **InternVL2_5-1B** | [`InternVL2_5-1B.ipynb`](InternVL2_5-1B.ipynb) | [OpenGVLab/InternVL2_5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SakibAhmedShuva/OCR-with-LLM-Models/blob/main/InternVL2_5-1B.ipynb) |
| **InternVL2_5-2B** | [`InternVL2_5-2B.ipynb`](InternVL2_5-2B.ipynb) | [OpenGVLab/InternVL2_5-2B](https://huggingface.co/OpenGVLab/InternVL2_5-2B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SakibAhmedShuva/OCR-with-LLM-Models/blob/main/InternVL2_5-2B.ipynb) |
| **MiniCPM-V-2.6 (INT4)** | [`MiniCPM-V-2_6-int4.ipynb`](MiniCPM-V-2_6-int4.ipynb) | [openbmb/MiniCPM-V-2_6-int4](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SakibAhmedShuva/OCR-with-LLM-Models/blob/main/MiniCPM-V-2_6-int4.ipynb) |
| **Phi-3.5 Vision Instruct** | [`Phi3.5-Vision.ipynb`](Phi3.5-Vision.ipynb) | [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SakibAhmedShuva/OCR-with-LLM-Models/blob/main/Phi3.5-Vision.ipynb) |
| **Qwen2.5-VL-7B Instruct (4-bit)** | [`Qwen2.5-VL-7B.ipynb`](Qwen2.5-VL-7B.ipynb) | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SakibAhmedShuva/OCR-with-LLM-Models/blob/main/Qwen2.5-VL-7B.ipynb) |
| **moondream2** | [`moondream2.ipynb`](moondream2.ipynb) | [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SakibAhmedShuva/OCR-with-LLM-Models/blob/main/moondream2.ipynb) |

## 🚀 Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SakibAhmedShuva/OCR-with-LLM-Models.git
   cd OCR-with-LLM-Models
   ```

2. **Create a Python environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers Pillow accelerate
   pip install bitsandbytes  # For 4-bit/8-bit quantization
   pip install decord        # For some vision model utilities
   pip install einops        # Often needed for vision transformers
   pip install qwen_vl_utils # Specific to Qwen-VL models
   # For Phi-3.5 Vision with OpenVINO:
   # pip install optimum[openvino]
   ```
   > **Note:** Specific notebooks might have slightly different dependencies. Check individual notebooks or use the "Open in Colab" links which handle dependencies automatically.

4. **CUDA (for GPU acceleration):**
   Ensure you have a compatible CUDA toolkit installed if you plan to run these models on a GPU. The notebooks are generally configured to leverage GPUs if available (e.g., T4 on Colab).

## 📋 Usage

Each Jupyter Notebook (`.ipynb` file) in this repository is a self-contained example for a specific model.

1. **Open a notebook:**
   * Locally using Jupyter Lab/Notebook
   * Or click the "Open In Colab" badge next to the model you're interested in

2. **Image Path:**
   The notebooks typically use a sample image: `California -USA-_Front_f4dce8c3dfba3c893adafeec60cdd00d.jpg`
   * If running locally, ensure this image (or your own) is in the correct path specified within the notebook
   * You may need to upload your desired image to the Colab environment if running there

3. **Run the cells:**
   Execute the cells in the notebook sequentially. The cells will:
   * Install necessary packages (if any specific to the notebook)
   * Load the pre-trained model and tokenizer/processor
   * Load and preprocess the image
   * Define the OCR prompt
   * Perform inference and print the extracted text

## 💬 Core OCR Prompt

The primary prompt used across these notebooks to instruct the LLMs for meticulous OCR is:

```
<image>
You are an expert Optical Character Recognition (OCR) assistant. Your sole and critical task is to meticulously extract ALL text, numbers, and symbols visible in the provided image. Transcribe the text exactly as it appears on the document, character by character. You MUST ONLY output the extracted text. Do NOT add any introductory phrases, explanations, summaries, confidence scores, meta-comments, or any text whatsoever that is not directly read from the image. Be extremely thorough and do not miss any textual element, no matter how small, faint, or seemingly insignificant.
```

*Minor variations of this prompt might be present in individual notebooks to suit specific model behaviors.*

## 📊 Sample Demonstration

Below is an example of the kind of output expected, using InternVL2_5-1B on the sample driver's license image:

<details>
<summary>Sample Extracted Text (click to expand)</summary>

```
I1234568
CALIFORNIA
DRIVER LICENSE
LN CARDHOLDER
FN IMA
2570 24TH STREET
SACRAMENTO, CA 95181
DOB 08/31/1977
DD 08/31/2010
EXP 08/31/2015
CLASS C
END NONE
0831977
SEX F
HAIR BRN
EYES BRN
DD 08/30/2010
ISS 09/30/2010
```

</details>

> Note: Actual output may vary slightly between models and runs.

## 🔮 Potential Improvements & Future Work

- **Unified Script:** Create a single Python script that can load and run any of the chosen models via command-line arguments
- **More Models:** Add demonstrations for other promising vision-language models
- **Performance Benchmarking:** Compare inference speed and resource usage across different models and hardware
- **Accuracy Evaluation:** Implement a method to evaluate OCR accuracy against ground truth for various document types
- **Advanced Preprocessing/Postprocessing:** Explore techniques to improve OCR quality for challenging images
- **Error Analysis:** Investigate common failure modes for each LLM in OCR tasks

## 🤝 Contributing

Contributions are welcome! If you'd like to add a new model, improve existing notebooks, or enhance the documentation, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

Please ensure your code adheres to good practices and that notebooks are clearly documented.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- The creators of the respective LLMs and Vision models
- Hugging Face for providing access to models and the transformers library
- The open-source community for their invaluable tools and resources
