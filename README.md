# AI-Driven Code Optimizer

The AI-Driven Code Optimizer is a Python script that analyzes and suggests optimizations for Python code. It combines techniques from static code analysis with machine learning to provide recommendations for improving code efficiency, readability, and adherence to best practices.

## Features

- **Code Analysis**: The script parses a Python file, identifies patterns and structures, and analyzes the usage of variables, functions, and loops using the Abstract Syntax Tree (AST).
- **Machine Learning Predictions**: It utilizes a pre-trained machine learning model to predict potential performance improvements or possible refactoring needs based on the analyzed code features.
- **Automatic Suggestions**: Based on the analysis and ML predictions, the script generates specific code change suggestions that can enhance performance or readability.
- **Report Generation**: It outputs a report detailing the suggested changes and their expected impact on code quality.
- **Error Handling**: Robust error handling to gracefully manage exceptions during model loading, code analysis, and optimization prediction.
- **Logging**: Comprehensive logging to provide detailed information about the script's execution and any issues encountered.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jayinmt/ai-code-optimizer.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Prepare the pre-trained machine learning model:
   - Train a machine learning model on a dataset of Python code snippets labeled with performance metrics.
   - Save the trained model as `model.pkl` in the project directory.

## Usage

1. Place the Python code file you want to optimize in the project directory.
2. Update the `main` function in `ai-code-optimizer.py` to specify the path to your code file:
   ```python
   with open('your_code_file.py', 'r') as file:
       code = file.read()
   ```
3. Run the script:
   ```
   python ai-code-optimizer.py
   ```
4. The script will analyze the code, generate optimization suggestions, and print the optimization report.
5. Check the log file for detailed information about the script's execution and any errors encountered.

## Technologies Used

- **Python**: The main programming language for the script.
- **AST (Abstract Syntax Tree)**: Used for parsing and analyzing the Python code.
- **Scikit-learn**: Used for implementing the Random Forest Classifier machine learning model.
- **NumPy/Pandas**: Used for data manipulation during the analysis phase.
- **Logging**: Python's built-in logging module for comprehensive error tracking and execution information.

## Error Handling

The script now includes robust error handling for common issues:
- File not found errors when loading the model or reading code files
- Syntax errors in the provided Python code
- Exceptions during model prediction or code analysis

All errors are logged with detailed information to aid in troubleshooting.

## Future Enhancements

- Improve the code analysis by considering more code patterns and best practices.
- Enhance the machine learning model with a larger and more diverse dataset.
- Provide interactive code refactoring suggestions within an IDE or code editor.
- Expand support for optimizing code written in other programming languages.
- Implement more advanced logging features, such as log rotation and remote logging.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The project was inspired by the need for automated code optimization tools.
- Thanks to the open-source community for providing valuable libraries and resources.
