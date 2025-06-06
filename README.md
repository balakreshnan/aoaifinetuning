# Azure OpenAI Fine-tuning Pipeline

A comprehensive Python-based solution for fine-tuning Azure OpenAI models with custom datasets. This repository demonstrates the complete workflow from data preparation to model deployment and testing.

## Overview

This project provides a complete pipeline for fine-tuning Azure OpenAI models, specifically designed to create custom chatbots with specialized behaviors. The included example demonstrates training a sarcastic chatbot called "Clippy" that provides factual information with humorous, sarcastic responses.

## Features

- **Data Validation**: Automated validation of training and validation datasets
- **Token Analysis**: Statistical analysis of token distribution in datasets
- **File Upload**: Seamless upload of datasets to Azure OpenAI
- **Job Management**: Automated fine-tuning job creation and monitoring
- **Model Deployment**: Streamlined deployment of fine-tuned models
- **Testing Framework**: Built-in testing capabilities for deployed models
- **Monitoring**: Real-time progress tracking with elapsed time reporting

## Prerequisites

### Azure Requirements

1. **Azure Subscription**: Active Azure subscription with sufficient credits
2. **Azure OpenAI Resource**: Provisioned Azure OpenAI service in a supported region
3. **Permissions**: Contributor access to the Azure OpenAI resource
4. **Authentication**: Service principal or managed identity for API access

### Software Requirements

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/balakreshnan/aoaifinetuning.git
   cd aoaifinetuning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Environment Variables

Set the following environment variables before running the script:

```bash
# Required for Azure OpenAI API access
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"

# Required for model deployment (if using deploymodel function)
export TEMP_AUTH_TOKEN="your-auth-token"
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_OPENAI_RESOURCE_NAME="your-openai-resource-name"
```

### Data Preparation

Ensure your training and validation datasets are in JSONL format with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "System message defining the assistant's behavior"},
    {"role": "user", "content": "User question or prompt"},
    {"role": "assistant", "content": "Expected assistant response"}
  ]
}
```

## Usage

### Basic Fine-tuning Workflow

1. **Prepare your datasets**: Place `training_set.jsonl` and `validation_set.jsonl` in the project root
2. **Set environment variables** as described in the Configuration section
3. **Run the fine-tuning pipeline**:
   ```bash
   python finetuneaoai.py
   ```

### Individual Function Usage

The script provides modular functions that can be used independently:

```python
from finetuneaoai import Checkdata, processtoken, uploadfinetunefiles, deploymodel, deploytest

# Validate datasets
Checkdata()

# Analyze token distribution
processtoken()

# Upload files to Azure OpenAI
uploadfinetunefiles()

# Deploy fine-tuned model (after fine-tuning is complete)
deploymodel()

# Test deployed model
deploytest()
```

## File Structure

```
aoaifinetuning/
├── README.md                 # This documentation file
├── finetuneaoai.py          # Main fine-tuning script
├── requirements.txt         # Python dependencies
├── training_set.jsonl       # Training dataset (example: sarcastic chatbot)
├── validation_set.jsonl     # Validation dataset
└── styles.css              # Streamlit styling (for web interface)
```

## Workflow Details

### 1. Data Validation (`Checkdata()`)
- Loads and validates training and validation datasets
- Displays dataset statistics and sample entries
- Ensures proper JSONL formatting

### 2. Token Analysis (`processtoken()`)
- Analyzes token distribution across datasets
- Provides statistical insights (min/max, mean/median, percentiles)
- Helps optimize training parameters

### 3. File Upload (`uploadfinetunefiles()`)
- Uploads datasets to Azure OpenAI service
- Returns file IDs for fine-tuning job creation
- Includes file processing status monitoring

### 4. Fine-tuning Job Management
- Creates fine-tuning jobs with specified parameters
- Monitors job progress with real-time status updates
- Provides detailed logging and error handling

### 5. Model Deployment (`deploymodel()`)
- Deploys fine-tuned models to Azure OpenAI endpoints
- Configures deployment parameters (SKU, capacity)
- Handles deployment status and error reporting

### 6. Model Testing (`deploytest()`)
- Tests deployed models with sample conversations
- Validates model behavior and response quality
- Provides example implementation for custom testing

## Example: Sarcastic Chatbot

The included datasets demonstrate training a chatbot with the following characteristics:

- **System Prompt**: "Clippy is a factual chatbot that is also sarcastic."
- **Behavior**: Provides accurate information with humorous, sarcastic tone
- **Use Case**: Educational tool with personality for engaging interactions

### Sample Training Data

```json
{
  "messages": [
    {"role": "system", "content": "Clippy is a factual chatbot that is also sarcastic."},
    {"role": "user", "content": "What is the largest planet?"},
    {"role": "assistant", "content": "It's called Jupiter, you might have heard of it...or not."}
  ]
}
```

## Monitoring and Logging

The script provides comprehensive monitoring features:

- **Real-time Progress**: Live updates during fine-tuning job execution
- **Elapsed Time Tracking**: Continuous time monitoring for job duration
- **Status Reporting**: Detailed status messages and error handling
- **Token Statistics**: In-depth analysis of dataset token usage

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Verify environment variables are set correctly
   - Check Azure OpenAI resource permissions
   - Ensure API keys are valid and not expired

2. **File Upload Failures**:
   - Confirm JSONL file formatting is correct
   - Check file sizes (Azure has upload limits)
   - Verify network connectivity to Azure services

3. **Fine-tuning Job Failures**:
   - Review training data quality and format
   - Check token limits and dataset size requirements
   - Monitor Azure service status and quotas

4. **Deployment Issues**:
   - Verify subscription and resource group names
   - Check deployment permissions and quotas
   - Ensure fine-tuning job completed successfully

### Debug Mode

Enable verbose logging by modifying the script to include additional print statements or by setting Python logging level to DEBUG.

## Best Practices

### Dataset Preparation
- **Quality over Quantity**: Focus on high-quality, diverse examples
- **Balanced Distribution**: Ensure representative coverage of use cases
- **Consistent Formatting**: Maintain consistent message structure and tone
- **Token Optimization**: Monitor token usage to avoid exceeding limits

### Fine-tuning Parameters
- **Model Selection**: Choose appropriate base models (e.g., gpt-4o-mini-2024-07-18)
- **Seed Setting**: Use consistent seed values for reproducible results
- **Monitoring**: Regularly check training progress and validation metrics

### Deployment Strategy
- **Testing**: Thoroughly test models before production deployment
- **Scaling**: Configure appropriate capacity for expected usage
- **Monitoring**: Implement ongoing monitoring for model performance

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

### Development Setup

1. Clone the repository
2. Install development dependencies: `pip install -r requirements.txt`
3. Set up pre-commit hooks for code quality
4. Write tests for new functionality
5. Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Resources

- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune)
- [OpenAI Fine-tuning Best Practices](https://platform.openai.com/docs/guides/fine-tuning)
- [Azure OpenAI Service Limits](https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Azure OpenAI service documentation
3. Submit issues through GitHub Issues
4. For Azure-specific problems, consult Azure support resources
