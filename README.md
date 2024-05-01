# Cpstone-project
{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NPrV9Hmy9TlV",
        "outputId": "cf40d6a3-227a-499e-ec57-d09aac016e0b"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NxTK3hMMDgfB",
        "outputId": "558c95dd-7a54-4687-d07f-77f4def881a3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[Errno 20] Not a directory: '/content/drive/MyDrive/Capstone Project- Medical Insurance Price Prediction/Medical_insurance.csv'\n",
            "/content\n"
          ]
        }
      ],
      "source": [
        "%cd /content/drive/MyDrive/Capstone Project- Medical Insurance Price Prediction/Medical_insurance.csv"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nfTDXbebrvcg",
        "outputId": "f44a570e-2b37-44eb-b5cc-8087c7df1f5f"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "drive  sample_data\n"
          ]
        }
      ],
      "source": [
        "!ls"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g-UYLPEhB6As"
      },
      "source": [
        "This Project is based on Medical Insurance Price Prediction available from kaggle repository.\n",
        "\n",
        "(https://www.kaggle.com/code/dylandsi/medical-insurance-price-prediction)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HA3R5155B7QZ"
      },
      "source": [
        "1. It contains the details (e,g, age, sex, bmi etc.)of 2773 customers.\n",
        "\n",
        "2. My project task is to create a machine learning model which can predict the price of insurance on its characteristics.\n",
        "\n",
        "3. For solving this problem, I will approach the task, with a step by step approach to create a data analysis and prediction model based on\n",
        "(machine learning/AI algorithms, regression algorith for example) available from different Python packages, modules and classes\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QOSQ8TFzCCXh"
      },
      "source": [
        "#**Step.1 : Reading the Dataset with Python**\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ik-eRP0BCP36"
      },
      "outputs": [],
      "source": [
        "# Supressing the warning messages\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Q0lsJXabCHG7"
      },
      "source": [
        " #**Data Loading and Inspection:**\n",
        "\n",
        "1. Load the dataset into a pandas DataFrame.\n",
        "2. Inspect the first few rows of the dataset to understand its structure and the type of data it contains.\n",
        "3. Check for missing values and handle them appropriately, either by imputing missing values or removing rows/columns with missing data if necessary.\n",
        "Check for any inconsistencies or anomalies in the data that may need to be addressed.\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 397
        },
        "id": "V1sO7apiCIHg",
        "outputId": "3f17ec02-41da-464f-ae07-195bedbeb289"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Shape before deleting duplicate values: (2302, 7)\n",
            "Shape After deleting duplicate values: (1112, 7)\n"
          ]
        },
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"insurance_data\",\n  \"rows\": 1112,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13,\n        \"min\": 18,\n        \"max\": 64,\n        \"num_unique_values\": 47,\n        \"samples\": [\n          34,\n          49,\n          53\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"male\",\n          \"female\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"bmi\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 5.840062285903609,\n        \"min\": 15.96,\n        \"max\": 45.9,\n        \"num_unique_values\": 499,\n        \"samples\": [\n          20.35,\n          19.95\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"children\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 5,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"smoker\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"no\",\n          \"yes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"region\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"southeast\",\n          \"northeast\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"charges\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 5584.666607908557,\n        \"min\": 1121.8739,\n        \"max\": 24227.33724,\n        \"num_unique_values\": 1112,\n        \"samples\": [\n          9861.025,\n          10579.711\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "insurance_data"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-7046aebd-b921-4ce7-b77c-fd037dd8afc9\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>bmi</th>\n",
              "      <th>children</th>\n",
              "      <th>smoker</th>\n",
              "      <th>region</th>\n",
              "      <th>charges</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>19</td>\n",
              "      <td>female</td>\n",
              "      <td>27.900</td>\n",
              "      <td>0</td>\n",
              "      <td>yes</td>\n",
              "      <td>southwest</td>\n",
              "      <td>16884.92400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>18</td>\n",
              "      <td>male</td>\n",
              "      <td>33.770</td>\n",
              "      <td>1</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "      <td>1725.55230</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>28</td>\n",
              "      <td>male</td>\n",
              "      <td>33.000</td>\n",
              "      <td>3</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "      <td>4449.46200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>33</td>\n",
              "      <td>male</td>\n",
              "      <td>22.705</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>northwest</td>\n",
              "      <td>21984.47061</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>32</td>\n",
              "      <td>male</td>\n",
              "      <td>28.880</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>northwest</td>\n",
              "      <td>3866.85520</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>31</td>\n",
              "      <td>female</td>\n",
              "      <td>25.740</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "      <td>3756.62160</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>46</td>\n",
              "      <td>female</td>\n",
              "      <td>33.440</td>\n",
              "      <td>1</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "      <td>8240.58960</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>37</td>\n",
              "      <td>female</td>\n",
              "      <td>27.740</td>\n",
              "      <td>3</td>\n",
              "      <td>no</td>\n",
              "      <td>northwest</td>\n",
              "      <td>7281.50560</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>37</td>\n",
              "      <td>male</td>\n",
              "      <td>29.830</td>\n",
              "      <td>2</td>\n",
              "      <td>no</td>\n",
              "      <td>northeast</td>\n",
              "      <td>6406.41070</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>25</td>\n",
              "      <td>male</td>\n",
              "      <td>26.220</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>northeast</td>\n",
              "      <td>2721.32080</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7046aebd-b921-4ce7-b77c-fd037dd8afc9')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-7046aebd-b921-4ce7-b77c-fd037dd8afc9 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-7046aebd-b921-4ce7-b77c-fd037dd8afc9');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-f5ffedbd-f64f-490d-b5ac-4d8dae6093f3\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-f5ffedbd-f64f-490d-b5ac-4d8dae6093f3')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-f5ffedbd-f64f-490d-b5ac-4d8dae6093f3 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "   age     sex     bmi  children smoker     region      charges\n",
              "0   19  female  27.900         0    yes  southwest  16884.92400\n",
              "1   18    male  33.770         1     no  southeast   1725.55230\n",
              "2   28    male  33.000         3     no  southeast   4449.46200\n",
              "3   33    male  22.705         0     no  northwest  21984.47061\n",
              "4   32    male  28.880         0     no  northwest   3866.85520\n",
              "5   31  female  25.740         0     no  southeast   3756.62160\n",
              "6   46  female  33.440         1     no  southeast   8240.58960\n",
              "7   37  female  27.740         3     no  northwest   7281.50560\n",
              "8   37    male  29.830         2     no  northeast   6406.41070\n",
              "9   25    male  26.220         0     no  northeast   2721.32080"
            ]
          },
          "execution_count": 5,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Reading the dataset\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "insurance_data=pd.read_csv('/content/drive/MyDrive/Capstone Project- Medical Insurance Price Prediction/Medical_insurance.csv', encoding='latin')\n",
        "print('Shape before deleting duplicate values:', insurance_data.shape)\n",
        "\n",
        "# Removing duplicate rows if any\n",
        "insurance_data=insurance_data.drop_duplicates()\n",
        "print('Shape After deleting duplicate values:', insurance_data.shape)\n",
        "\n",
        "# Printing sample data\n",
        "# Start observing the Quantitative/Categorical/Qualitative variables\n",
        "insurance_data.head(10)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WwxpI_vrGd5c"
      },
      "source": [
        "#**Step.2 :Problem Statement Definition**\n",
        "\n",
        "Develop a machine learning model to predict medical expenses for individuals based on various factors such as age, gender, BMI, number of children, smoking status, and region. The model should provide accurate estimates of medical charges for new customers, enabling the insurance company to make informed decisions regarding pricing and risk assessment."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6myV-va4Hl22"
      },
      "source": [
        "#**Step.3 : Target Variable Identification**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "q7rg8PBHFvQI",
        "outputId": "92381197-4d58-4c8e-fac2-6ed9b6cb0958"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Index(['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'], dtype='object')\n",
            "   age     sex     bmi  children smoker     region      charges\n",
            "0   19  female  27.900         0    yes  southwest  16884.92400\n",
            "1   18    male  33.770         1     no  southeast   1725.55230\n",
            "2   28    male  33.000         3     no  southeast   4449.46200\n",
            "3   33    male  22.705         0     no  northwest  21984.47061\n",
            "4   32    male  28.880         0     no  northwest   3866.85520\n",
            "0       16884.92400\n",
            "1        1725.55230\n",
            "2        4449.46200\n",
            "3       21984.47061\n",
            "4        3866.85520\n",
            "           ...     \n",
            "2297     8569.86180\n",
            "2298     2020.17700\n",
            "2299    16450.89470\n",
            "2300    21595.38229\n",
            "2301     9850.43200\n",
            "Name: charges, Length: 2302, dtype: float64\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Load the medical insurance dataset\n",
        "insurance_data = pd.read_csv('/content/drive/MyDrive/Capstone Project- Medical Insurance Price Prediction/Medical_insurance.csv')\n",
        "\n",
        "# Check column names\n",
        "print(insurance_data.columns)\n",
        "\n",
        "# Inspect the dataset\n",
        "print(insurance_data.head())\n",
        "\n",
        "# Extract the target variable if the column name is correct\n",
        "if 'charges' in insurance_data.columns:\n",
        "    target_variable = insurance_data['charges']\n",
        "    print(target_variable)\n",
        "else:\n",
        "    print(\"Column 'charges' not found in the dataset.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jtEnQEStGIKK"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Set the style of seaborn\n",
        "sns.set(style=\"whitegrid\")\n",
        "\n",
        "# Create a figure and axis object\n",
        "plt.figure(figsize=(10, 6))\n",
        "\n",
        "# Plot the distribution of charges\n",
        "sns.histplot(insurance_data['charges'], kde=True, color='skyblue')\n",
        "\n",
        "# Add labels and title\n",
        "plt.title('Distribution of Medical Charges')\n",
        "plt.xlabel('Charges')\n",
        "plt.ylabel('Frequency')\n",
        "\n",
        "# Show the plot\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WP7qTdeZHvuB"
      },
      "source": [
        "#**Step 4: Choosing the appropriate ML/AI Algorithm for Data Analysis.**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fh08rft4HRDV"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "\n",
        "# Load the dataset\n",
        "insurance_data = pd.read_csv('/content/drive/MyDrive/Medical_insurance.csv')\n",
        "\n",
        "# Perform one-hot encoding for categorical variables\n",
        "insurance_data_encoded = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'])\n",
        "\n",
        "# Prepare the data\n",
        "X = insurance_data_encoded.drop(columns=['charges'])\n",
        "y = insurance_data_encoded['charges']\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Initialize the Random Forest Regression model\n",
        "model = RandomForestRegressor(random_state=42)\n",
        "\n",
        "# Train the model\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Make predictions on the testing data\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Evaluate the model\n",
        "mae = mean_absolute_error(y_test, y_pred)\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "\n",
        "print(\"Mean Absolute Error:\", mae)\n",
        "print(\"Mean Squared Error:\", mse)\n",
        "print(\"R-squared:\", r2)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2UusrZRaJRAZ"
      },
      "source": [
        "#**Step.5 :Looking at the class distribution (Target variable distribution to check if the data is balanced or skewed.**\n",
        "* If target variable's distribution is too skewed then the predictive modeling will lead to poor results.\n",
        "* Ideally Bell curve is desirable but slightly positive skew or negative skew is also fine.\n",
        "* When performing Regression algorithm modelling and analysis, we need to make sure the histogram looks like a bell curve or slight skewed version of it.\n",
        "* Otherwise it impacts the Machine Learning algorithms ability to learn all the scenarios from the data."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "u7tZKWdsKfOR"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Assuming you have loaded the medical insurance dataset into a variable named 'insurance_data'\n",
        "# Replace 'insurance_data' with the appropriate variable name if it's different\n",
        "insurance_data['charges'].hist()\n",
        "\n",
        "plt.title('Distribution of Medical Expenses (Charges)')\n",
        "plt.xlabel('Charges')\n",
        "plt.ylabel('Frequency')\n",
        "\n",
        "plt.show()\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Nvq-E6QQ8OFZ"
      },
      "outputs": [],
      "source": [
        "%matplotlib inline\n",
        "# Creating histogram as the Target variable is Continuous\n",
        "# This will help us to understand the distribution of the MEDV values\n",
        "insurance_data['charges'].hist()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PE6M3t2fIJVY"
      },
      "source": [
        "#**Step.6 :Visualising the distribution of Target variable**\n",
        "\n",
        "\n",
        "\n",
        "This code will create a histogram showing the distribution of medical expenses (charges) among the individuals in the dataset. Adjust the number of bins and colors as needed to improve the visualization."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zN5Pi1X3N09u"
      },
      "outputs": [],
      "source": [
        "# Importing necessary libraries\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Load the medical insurance dataset\n",
        "insurance_data = pd.read_csv('/content/drive/MyDrive/Capstone Project- Medical Insurance Price Prediction/Medical_insurance.csv')\n",
        "\n",
        "# Define numerical and categorical columns\n",
        "numerical_columns = ['age', 'bmi', 'children', 'charges']\n",
        "categorical_columns = ['sex', 'smoker', 'region']\n",
        "\n",
        "# Explore the distributions of variables through visualizations\n",
        "# Histograms for numerical variables\n",
        "insurance_data[numerical_columns].hist(bins=20, figsize=(15, 10))\n",
        "plt.suptitle('Histograms of Numerical Variables', fontsize=16)\n",
        "plt.show()\n",
        "\n",
        "# Box plots for numerical variables\n",
        "plt.figure(figsize=(15, 10))\n",
        "sns.boxplot(data=insurance_data[numerical_columns])\n",
        "plt.title('Box Plots of Numerical Variables', fontsize=16)\n",
        "plt.show()\n",
        "\n",
        "# Scatter plot for BMI vs. Charges\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.scatterplot(x='bmi', y='charges', data=insurance_data)\n",
        "plt.title('Scatter Plot of BMI vs. Charges', fontsize=16)\n",
        "plt.xlabel('BMI')\n",
        "plt.ylabel('Charges')\n",
        "plt.show()\n",
        "\n",
        "# Investigate correlations between numerical variables\n",
        "correlation_matrix = insurance_data[numerical_columns].corr()\n",
        "plt.figure(figsize=(12, 8))\n",
        "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=\".2f\")\n",
        "plt.title('Correlation Matrix of Numerical Variables', fontsize=16)\n",
        "plt.show()\n",
        "\n",
        "# Box plots for numerical and categorical variables\n",
        "plt.figure(figsize=(15, 10))\n",
        "sns.boxplot(data=insurance_data[numerical_columns + categorical_columns])\n",
        "plt.title('Box Plots of Numerical and Categorical Variables', fontsize=16)\n",
        "plt.show()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_Fiz5vpQNNOy"
      },
      "source": [
        "## Observations from previous step\n",
        "* The data distribution of the target variable is satisfactory to proceed further.\n",
        "* There are sufficient number of rows for each type of values to learn from."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kc_BV80VNnPz"
      },
      "source": [
        "## **Step.7: Basic Exploratory Data Analysis**\n",
        "* This step is performed to guage the overall data.\n",
        "* The volume of data, the types of columns present in the data.\n",
        "* Initial assessment of the data should be done to identify which columns are Quantitative, Categorical or Qualitative.\n",
        "\n",
        "* This step helps to start the column/data rejection process.\n",
        "* You must look at each column carefully and ask, does this column affect the values of the Target variable/Class?\n",
        "* For example in this dataset, you will ask, does this column affect the price of the house?\n",
        "* If the answer is a clear \"No\", then remove the column immediately from the data, otherwise keep the column for further analysis.\n",
        "\n",
        "* There are four commands which are used for Basic data exploratory Analysis in Python\n",
        "\n",
        "* head() : This helps to see a few sample rows of the data\n",
        "* info() : This provides the summarized information of the data\n",
        "* describe() : This provides the descriptive statistical details of the data\n",
        "* nunique(): This helps us to identify if a column is categorical or continuous\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8iArNhsVMLwb"
      },
      "outputs": [],
      "source": [
        "# Looking at sample rows in the data\n",
        "insurance_data.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EXs7WogROoC3"
      },
      "outputs": [],
      "source": [
        "# Looking at sample rows in the data\n",
        "insurance_data.tail()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "g9_nGRIsPEuz"
      },
      "outputs": [],
      "source": [
        "# Observing the summarized information of data\n",
        "# Data types, Missing values based on number of non-null values Vs total rows etc.\n",
        "# Remove those variables from data which have too many missing values (Missing Values > 30%)\n",
        "# Remove Qualitative variables which cannot be used in Machine Learning\n",
        "insurance_data.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "WpuYNsk9PLQI"
      },
      "outputs": [],
      "source": [
        "# Looking at the descriptive statistics of the data\n",
        "insurance_data.describe(include='all')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s7JWq_QyPUZi",
        "outputId": "ac874a53-03f2-4ec6-c16d-1328020148ef"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "age           47\n",
              "sex            2\n",
              "bmi          499\n",
              "children       6\n",
              "smoker         2\n",
              "region         4\n",
              "charges     1112\n",
              "dtype: int64"
            ]
          },
          "execution_count": 128,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Finging unique values for each column\n",
        "# TO understand which column is categorical and which one is Continuous\n",
        "# Typically if the numer of unique values are < 20 then the variable is likely to be a category otherwise continuous\n",
        "insurance_data.nunique()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RPlhmTJGUXMX"
      },
      "source": [
        "## **Observations from Step.7 - Basic Exploratory Data Analysis**\n",
        "* Based on the basic exploration above, you can now create a simple report of the data, noting down your observations regaring each column.\n",
        "* Hence, creating a initial roadmap for further analysis.\n",
        "\n",
        "* The selected columns in this step are not final, further study will be done and then a final list will be created\n",
        "\n",
        "\n",
        "* age - Continuous. Selected.\n",
        "* sex - Categorical. Selected.\n",
        "* bmi - Continuous. Selected.\n",
        "* children - Categorical. Selected.\n",
        "* smoker - Categorical. Selected.\n",
        "* region - Categorical. Selected.\n",
        "* charges - Continuous. Selected.\n",
        " This is the Target or Class Variable, which needs to be predicted by the proposed regression model!"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6LChxHPxTt00"
      },
      "source": [
        "#**Step.8 :Identifying and Rejecting useless columns**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yrTJYlsNUI8m"
      },
      "source": [
        "1. There are no qualitative columns in the data.\n",
        "2. Hence no need to remove any column."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4MjHwBGwZcki"
      },
      "source": [
        "#**Step.9 :Visual Exploratory Data Analysis of data (with Histogramand Barcharts)**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-vapnKU5aRuB"
      },
      "source": [
        "* Visualize distribution of all the Categorical Predictor variables in the data using bar plots\n",
        "* We can spot a categorical variable in the data by looking at the unique values in them.\n",
        "* Typically a categorical variable contains less than 20 Unique values AND there is repetition of values, which means the data can be grouped by those unique values.\n",
        "* Based on the Basic Exploration Data Analysis in the previous step,  we could spotted two categorical predictors in the data\n",
        "\n",
        "* Categorical Predictors:\n",
        "\n",
        "* 'sex'\n",
        "* 'children'\n",
        "* 'smoker'\n",
        "* 'region'\n",
        "\n",
        "* We will use bar charts to see how the data is distributed for these categorical columns."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "UV_u7aQFZukS"
      },
      "outputs": [],
      "source": [
        "# Plotting multiple bar charts at once for categorical variables\n",
        "# Since there is no default function which can plot bar charts for multiple columns at once\n",
        "# we are defining our own function for the same\n",
        "\n",
        "def PlotBarCharts(inpData, colsToPlot):\n",
        "    %matplotlib inline\n",
        "\n",
        "    import matplotlib.pyplot as plt\n",
        "\n",
        "    # Generating multiple subplots\n",
        "    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(20,5))\n",
        "    fig.suptitle('Bar charts of: '+ str(colsToPlot))\n",
        "\n",
        "    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):\n",
        "        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 385
        },
        "id": "MwEDSCbbZ4bj",
        "outputId": "db97239a-4828-4070-ed69-7e5996f15960"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABk4AAAIrCAYAAACpuWWeAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACtnUlEQVR4nOzde1xVVf7/8fcBBBQ5ICYcmRAZp1TMW+ogeUmTL4hoOjE1JqUWSRlYamPGd4xRtDArr5FmmdqEXxtnyooURS2tRDSMNPTLWGlYdmC+o3DEkouc3x8d9s+Td+Pi5fV8PPYj9lqfvddahwMn+ey1lslut9sFAAAAAAAAAAAAuTR2BwAAAAAAAAAAAK4UJE4AAAAAAAAAAAAcSJwAAAAAAAAAAAA4kDgBAAAAAAAAAABwIHECAAAAAAAAAADgQOIEAAAAAAAAAADAgcQJAAAAAAAAAACAA4kTAAAAAAAAAAAABxInAAAAAAAAAAAADiROAAAAoAEDBuiWW25p7G40iKysLHXr1k2enp4ymUwqLS296GunT58uk8kkk8mk5s2b118nryJjx4696NfCZDJp+vTpxvmKFStkMpl06NChC17btm1bjR079vI6iXM6dOiQTCaTXnjhhcbuyhVt7Nixatu2bYO3261bN+N3ztChQxu8fQAAgOsViRMAAIB6UvtH4dMPf39/DRw4UOvXr2/s7jWaI0eOaPr06crPz2/wtv/zn//onnvuUdOmTZWenq6//e1v8vLyuuT7/O1vf9OyZcucygYMGNDof9ifPn36Zf9x96OPPrroJMb1yGQyacWKFZd17ZXw3rgS/JrX4de8t68WtZ8Zp3v22Wf1t7/9TTfccEMj9QoAAOD65NbYHQAAALjWpaamKiQkRHa7XcXFxVqxYoWGDBmi999//7p8gvjIkSOaMWOG2rZtq27dujVo27t27dLx48c1c+ZMRUREXPZ97rvvvjrs1fXjp59+kpsb/wTB1efVV19VTU1Ng7c7ZMgQSdK0adMavG0AAIDrGf9qAQAAqGfR0dHq2bOncR4fH6+AgAD9z//8T50kTmpqalRZWSlPT89ffa/6VF1d3Sh/eDxdSUmJJMnX17dR+3G9qu/36IkTJy5rBhHq148//qhmzZrVezv1+buwSZMmdX5PAAAAXLlYqgsAAKCB+fr6qmnTpmc8ef/CCy/otttuU8uWLdW0aVP16NFD//jHP8643mQyKSkpSRkZGerUqZM8PDyUlZV13jbXr1+v22+/Xd7e3jKbzerVq5dWrVp1Rty+ffs0cOBANWvWTL/5zW80Z84cp/rKykqlpKSoR48e8vHxkZeXl/r166cPP/zQKe70fRPmz5+vdu3aycPDQy+//LJ69eolSXrggQeMJcxql0A6cOCAYmNjZbFY5OnpqRtvvFEjR45UWVnZBV/XNWvWqEePHmratKluuOEG3Xffffr++++N+gEDBmjMmDGSpF69eslkMhnLBn399df6+uuvL9jGpVq0aJE6deqkZs2aqUWLFurZs+cZr/v333+vBx98UAEBAfLw8FCnTp30+uuvG/U//fSTOnTooA4dOuinn34yyo8eParWrVvrtttu06lTp+q877Vyc3M1ZMgQtWjRQl5eXurSpYsWLFhwRtz333+vESNGqHnz5mrVqpX+/Oc/n9GvX+5xcjZ2u12zZs3SjTfeqGbNmmngwIEqKCg4I652WaOtW7fq0Ucflb+/v2688Uajfv369erXr5+8vLzk7e2tmJiYM+5Tuz/LxfS9rl3ovVG7n86//vUv3XffffLx8VGrVq309NNPy2636/Dhwxo+fLjMZrMsFotefPHFM9ooKSkxErWenp7q2rWrVq5cecG+2e12JSQkyN3dXW+//bZR/uabbxo/Y35+fho5cqQOHz7sdG3tfkl5eXnq37+/mjVrpv/+7//+Fa/UuZ3vd+GFfq5qffvtt7rzzjvl5eUlf39/TZo0SRs2bJDJZNJHH31kxJ1tj5MTJ07oiSeeUFBQkDw8PNS+fXu98MILstvtZ+3n2rVrdcsttxj9udDvbQAAADQeZpwAAADUs7KyMv3f//2f7Ha7SkpKtGjRIpWXl5+x3NOCBQt05513Ki4uTpWVlVq9erXuvvtuZWZmKiYmxil2y5Yt+vvf/66kpCTdcMMN5137f8WKFXrwwQfVqVMnJScny9fXV59//rmysrI0atQoI+7YsWMaPHiw7rrrLt1zzz36xz/+oalTp6pz586Kjo6WJNlsNr322mu69957NW7cOB0/flzLli1TVFSUdu7cecbSW8uXL9fJkyeVkJAgDw8P/eEPf9Dx48eVkpKihIQE9evXT5J02223qbKyUlFRUaqoqNCECRNksVj0/fffKzMzU6WlpfLx8TnvGB944AH16tVLaWlpKi4u1oIFC/Tpp5/q888/l6+vr/7yl7+offv2Wrp0qbF8Wrt27SRJgwYNkqQ63d/j1Vdf1WOPPaY//vGPevzxx3Xy5Ent2bNHubm5xuteXFys3r17G39YbdWqldavX6/4+HjZbDZNnDhRTZs21cqVK9WnTx/95S9/0dy5cyVJiYmJKisr04oVK+Tq6lpn/T5ddna2hg4dqtatW+vxxx+XxWLR/v37lZmZqccff9yIO3XqlKKiohQWFqYXXnhBmzZt0osvvqh27dpp/Pjxl9RmSkqKZs2apSFDhmjIkCHavXu3IiMjVVlZedb4Rx99VK1atVJKSopOnDgh6ec9aMaMGaOoqCg999xz+vHHH7V48WL17dtXn3/+udPPS132/WJdzHuj1p/+9Cd17NhRs2fP1gcffKBZs2bJz89Pr7zyiu644w4999xzysjI0J///Gf16tVL/fv3l/Rzwm3AgAH66quvlJSUpJCQEK1Zs0Zjx45VaWmp0/fvdKdOndKDDz6ot956S++8847xu+eZZ57R008/rXvuuUcPPfSQ/v3vf2vRokXq37+/8TNW6z//+Y+io6M1cuRI3XfffQoICKiX11E6++/Ci/m5kn5OfNxxxx364YcfjPf3qlWrzkgEn43dbtedd96pDz/8UPHx8erWrZs2bNigKVOm6Pvvv9e8efOc4j/55BO9/fbbevTRR+Xt7a2FCxcqNjZWRUVFatmyZX28NAAAAPg17AAAAKgXy5cvt0s64/Dw8LCvWLHijPgff/zR6byystJ+yy232O+44w6nckl2FxcXe0FBwQX7UFpaavf29raHhYXZf/rpJ6e6mpoa4+vbb7/dLsn+xhtvGGUVFRV2i8Vij42NNcqqq6vtFRUVTvc5duyYPSAgwP7ggw8aZQcPHrRLspvNZntJSYlT/K5du+yS7MuXL3cq//zzz+2S7GvWrLnguE5XWVlp9/f3t99yyy1OY8zMzLRLsqekpBhltd+TXbt2Od0jODjYHhwcfMG2/vrXv9ov9n+hhw8fbu/UqdN5Y+Lj4+2tW7e2/9///Z9T+ciRI+0+Pj5O74nk5GS7i4uLfdu2bfY1a9bYJdnnz59/UX25HNXV1faQkBB7cHCw/dixY051p793xowZY5dkT01NdYrp3r27vUePHk5lkux//etfjfPa78fBgwftdrvdXlJSYnd3d7fHxMQ4tfHf//3fdkn2MWPGnHFt37597dXV1Ub58ePH7b6+vvZx48Y5tW21Wu0+Pj5O5ZfS97p0Me+N2vdaQkKCUVZdXW2/8cYb7SaTyT579myj/NixY/amTZs6vT7z58+3S7K/+eabRlllZaU9PDzc3rx5c7vNZrPb7f//Z/X555+3V1VV2f/0pz/ZmzZtat+wYYNx3aFDh+yurq72Z555xqmPe/futbu5uTmV1/4uWbJkyaW9KJfhXL8LL/bn6sUXX7RLsq9du9aI+emnn+wdOnSwS7J/+OGHRvmYMWOcfkesXbvWLsk+a9Yspzb++Mc/2k0mk/2rr75y6qe7u7tT2RdffGGXZF+0aNFFjTU4ONgeExNzUbEAAAD49ViqCwAAoJ6lp6crOztb2dnZevPNNzVw4EA99NBDTkvgSFLTpk2Nr48dO6aysjL169dPu3fvPuOet99+u0JDQy/YdnZ2to4fP66nnnrqjHX/TSaT03nz5s2dZsG4u7vr97//vb755hujzNXVVe7u7pJ+3k/g6NGjqq6uVs+ePc/az9jYWLVq1eqC/ZRkzCjZsGGDfvzxx4u6RpI+++wzlZSU6NFHH3UaY0xMjDp06KAPPvjggvc4dOhQnc42kX5eku27777Trl27zlpvt9v1z3/+U8OGDZPdbtf//d//GUdUVJTKysqcXtPp06erU6dOGjNmjB599FHdfvvteuyxx+q0z6f7/PPPdfDgQU2cOPGMPWF++d6RpEceecTpvF+/fk7vnYuxadMmVVZWasKECU5t1M4QOJtx48Y5zbjJzs5WaWmp7r33XqfX1NXVVWFhYWedTVAXfb8UF3pvnO6hhx4yvnZ1dVXPnj1lt9sVHx/vdL/27ds79XndunWyWCy69957jbImTZroscceU3l5ubZu3erUTmVlpTHDbd26dYqMjDTq3n77bdXU1Oiee+5xek0tFotuuummM15TDw8PPfDAAxf/gvwKv/xdeCk/V1lZWfrNb36jO++807je09NT48aNu2C769atk6ur6xk/g0888YTsdrvWr1/vVB4REWHMcJOkLl26yGw21+v7DAAAAJePpboAAADq2e9//3unzeHvvfdede/eXUlJSRo6dKiRiMjMzNSsWbOUn5+viooKI/5sf6QOCQm5qLZr9+245ZZbLhh74403ntFWixYttGfPHqeylStX6sUXX9T//u//qqqq6rx9uth+1sZOnjxZc+fOVUZGhvr166c777zT2N/hXL799ltJUvv27c+o69Chgz755JOL7kNdmjp1qjZt2qTf//73+t3vfqfIyEiNGjVKffr0kST9+9//VmlpqZYuXaqlS5ee9R61m9lLPyeyXn/9dfXq1Uuenp5avnz5Wd8bdeVS3juenp5nJMhatGihY8eOXVKbtd/Lm266yam8VatWatGixVmv+eV77MCBA5KkO+6446zxZrPZ6byu+n4pLvTeOF2bNm2czn18fOTp6akbbrjhjPL//Oc/xvm3336rm266SS4uzs/KdezY0ag/XVpamsrLy7V+/XoNGDDAqe7AgQOy2+1nfF9q/XLj9N/85jfG77X69svv/6X8XH377bdq167dGT9Hv/vd7y7Y7rfffqvAwEB5e3s7lZ/r9f3l91Gq//cZAAAALh+JEwAAgAbm4uKigQMHasGCBTpw4IA6deqkjz/+WHfeeaf69++vl19+Wa1bt1aTJk20fPnys27ifvrslLpyrn0y7KdtdPzmm29q7NixGjFihKZMmSJ/f3+5uroqLS3trJurX2o/X3zxRY0dO1bvvvuuNm7cqMcee0xpaWnasWOH08bfV4OOHTuqsLBQmZmZysrK0j//+U+9/PLLSklJ0YwZM1RTUyNJuu+++4xN63+pS5cuTucbNmyQJJ08eVIHDhy4pMRUfaqvPVYuxi/fY7Wv69/+9jdZLJYz4t3cnP8J1Bh9v9B740L9u5if1UsVFRWlrKwszZkzRwMGDHCavVVTUyOTyaT169efte3mzZs7ndfH76dzOdf3/1J+rhpCfXzPAAAAUH9InAAAADSC6upqSVJ5ebkk6Z///Kc8PT21YcMGeXh4GHHLly//Ve3ULg3z5ZdfXtRT1Bfyj3/8Q7/97W/19ttvOz2l/de//vWi73GhWRKdO3dW586dNW3aNG3fvl19+vTRkiVLNGvWrLPGBwcHS5IKCwvPmGVQWFho1DcGLy8v/elPf9Kf/vQnVVZW6q677tIzzzyj5ORktWrVSt7e3jp16pQiIiIueK89e/YoNTVVDzzwgPLz8/XQQw9p7969552N82uc/t65mP7Vhdrv1YEDB/Tb3/7WKP/3v/990U/m1/bb39+/wfp9Oc733vjlsnqXIzg4WHv27FFNTY3TrJP//d//NepP17t3bz3yyCMaOnSo7r77br3zzjtGkqldu3ay2+0KCQnRzTff/Kv7Vp8u5ecqODhY+/btk91ud/q99NVXX12wneDgYG3atEnHjx93mnVyrtcXAAAAVxf2OAEAAGhgVVVV2rhxo9zd3Y1lXVxdXWUymXTq1Ckj7tChQ1q7du2vaisyMlLe3t5KS0vTyZMnneou50nn2qemT782NzdXOTk5F30PLy8vSVJpaalTuc1mMxJKtTp37iwXFxenpct+qWfPnvL399eSJUuc4tavX6/9+/crJibmgn36+uuvzzpj5tc4fdkk6eeltkJDQ2W321VVVSVXV1fFxsbqn//8p7788sszrv/3v/9tfF1VVaWxY8cqMDBQCxYs0IoVK1RcXKxJkybVaZ9Pd+uttyokJETz588/43tVX0/JR0REqEmTJlq0aJFTG/Pnz7/oe0RFRclsNuvZZ591Wkqu1umva2O50HujLgwZMkRWq1VvvfWWUVZdXa1FixapefPmuv3228+4JiIiQqtXr1ZWVpbuv/9+Y/bGXXfdJVdXV82YMeOM773dbj9jPI3pUn6uoqKi9P333+u9994zyk6ePKlXX331gu0MGTJEp06d0ksvveRUPm/ePJlMJkVHR/+KUQAAAKCxMeMEAACgnq1fv954CrmkpESrVq3SgQMH9NRTTxn7LcTExGju3LkaPHiwRo0apZKSEqWnp+t3v/vdGXuMXAqz2ax58+bpoYceUq9evTRq1Ci1aNFCX3zxhX788UetXLnyku43dOhQvf322/rDH/6gmJgYHTx4UEuWLFFoaKgxe+ZC2rVrJ19fXy1ZskTe3t7y8vJSWFiYvvjiCyUlJenuu+/WzTffrOrqav3tb38z/hB6Lk2aNNFzzz2nBx54QLfffrvuvfdeFRcXa8GCBWrbtu1FJRcGDRokSXW6QXxkZKQsFov69OmjgIAA7d+/Xy+99JJiYmKMJ9Rnz56tDz/8UGFhYRo3bpxCQ0N19OhR7d69W5s2bdLRo0clydj7ZvPmzfL29laXLl2UkpKiadOm6Y9//KOGDBlyzn5Mnz5dM2bM0IcffnjG3hXn4+LiosWLF2vYsGHq1q2bHnjgAbVu3Vr/+7//q4KCAmPZsLrUqlUr/fnPf1ZaWpqGDh2qIUOG6PPPP9f69evP2NPjXMxmsxYvXqz7779ft956q0aOHKlWrVqpqKhIH3zwgfr06XPGH7svl8lk0u23366PPvrokq67mPfGr5WQkKBXXnlFY8eOVV5entq2bat//OMf+vTTTzV//vxztjNixAgtX75co0ePltls1iuvvKJ27dpp1qxZSk5O1qFDhzRixAh5e3vr4MGDeuedd5SQkKA///nPl9XPAQMGaOvWrXWajLvYn6uHH35YL730ku699149/vjjat26tTIyMowZP+ebHTds2DANHDhQf/nLX3To0CF17dpVGzdu1LvvvquJEyc6bQQPAACAqw+JEwAAgHqWkpJifO3p6akOHTpo8eLFevjhh43yO+64Q8uWLdPs2bM1ceJEhYSE6LnnntOhQ4d+VeJEkuLj4+Xv76/Zs2dr5syZatKkiTp06HBZsxXGjh0rq9WqV155RRs2bFBoaKjefPNNrVmz5qL/eNykSROtXLlSycnJeuSRR1RdXa3ly5fr9ttvV1RUlN5//319//33atasmbp27ar169erd+/eF+xXs2bNNHv2bE2dOlVeXl76wx/+oOeee06+vr6XPM668PDDDysjI0Nz585VeXm5brzxRj322GOaNm2aERMQEKCdO3cqNTVVb7/9tl5++WW1bNlSnTp10nPPPSdJ2r17t5599lklJSVp4MCBxrVPPfWU3n33XY0bN04FBQXnHGd5eblMJtNZ9/u4kKioKH344YeaMWOGXnzxRdXU1Khdu3YaN27cJd/rYs2aNUuenp5asmSJ8cfvjRs3XtTMoVqjRo1SYGCgZs+ereeff14VFRX6zW9+o379+umBBx6ok37WJgpbt259yddezHvj12ratKk++ugjPfXUU1q5cqVsNpvat2+v5cuXa+zYsee99r777tPx48f16KOPymw26/nnn9dTTz2lm2++WfPmzTP2YQkKClJkZKTuvPPOy+5neXn5Zb03z+difq6kn/dm2bJliyZMmKAFCxaoefPmGj16tG677TbFxsaed8k0FxcXvffee0pJSdFbb72l5cuXq23btnr++ef1xBNP1Ol4AAAA0PBMdnajAwAAAC5K7eyNf//73zKZTGrZsmVjd+mCfv/73ys4OFhr1qxp7K5cU9atW6ehQ4fqiy++UOfOnRu7O1el48ePy8/PT/Pnz1diYmJjd8cwf/58TZo0Sd99951+85vfNGpfSktLVV1drVtvvVVdunRRZmZmo/YHAADg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UIcOHTRo0CANGTJEffv2dXpay8fHRxs3btTBgwfVo0cPPfHEE0pJSeEfJQAAAAAuW3FxsT744APFx8efUTd79my1bNlS3bt31/PPP6/q6mqjLicnR/3795e7u7tRFhUVpcLCQh07duyc7bEmPQBc28rLy/X111+rdevW6tGjh5o0aeK0CkthYaGKioqcVmHZu3ev0xL22dnZMpvNCg0NbfD+AwDO7ZJnnAwYMEB2u/2c9SaTSampqUpNTT1njJ+fn1atWnXedrp06aKPP/74UrsHAAAAAGe1cuVKeXt766677nIqf+yxx3TrrbfKz89P27dvV3Jysn744QfNnTtX0s/rzoeEhDhdc/qa9OdaTjgtLU0zZsyoh5EAABrDn//8Zw0bNkzBwcE6cuSI/vrXv8rV1VX33nuvfHx8FB8fr8mTJ8vPz09ms1kTJkxQeHi4evfuLUmKjIxUaGio7r//fs2ZM0dWq1XTpk1TYmIiM0oA4ApzyYkTAAAAALgavf7664qLi3OaDS/9PIu+VpcuXeTu7q6HH35YaWlpv+oPWcnJyU73ttlsbOgLAFex7777Tvfee6/+85//qFWrVurbt6927NihVq1aSZLmzZsnFxcXxcbGqqKiQlFRUXr55ZeN611dXZWZmanx48crPDxcXl5eGjNmzHkfPgYANA4SJwAAAACueR9//LEKCwv11ltvXTA2LCxM1dXVOnTokNq3by+LxWKsQV+LNekB4PqzevXq89Z7enoqPT1d6enp54wJDg7WunXr6rprAIA6dsl7nAAAAADA1WbZsmXq0aOHunbtesHY/Px8ubi4yN/fX9LPa9Jv27ZNVVVVRkx2drbat29/zmW6AAAAAFy9SJwAAAAAuGqVl5crPz9f+fn5kqSDBw8qPz9fRUVFRozNZtOaNWv00EMPnXF9Tk6O5s+fry+++ELffPONMjIyNGnSJN13331GUmTUqFFyd3dXfHy8CgoK9NZbb2nBggVOy3ABAAAAuHawVBcAAACAq9Znn32mgQMHGue1yYwxY8ZoxYoVkn5eWsVut+vee+8943oPDw+tXr1a06dPV0VFhUJCQjRp0iSnpIiPj482btyoxMRE9ejRQzfccINSUlKUkJBQv4MDAAAA0ChInAAAAAC4ag0YMEB2u/28MQkJCedMctx6663asWPHBdvp0qWLPv7448vqIwAA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2bNnq0qVKmrSpInR8QAAAADgkWIymdS4cWM1btxYw4cPl7e3t7777rsH+qxatWppxowZysjIMBc31q9fLzs7O9WsWdP8Ond3dy1atEghISHq2LGjvv32WxUtWlT169fXqVOnVKRIEfn4+Fji8ADYAH9/f509e9boGDajW7duzODJp8mTJ5v/XK1aNQ0bNkw///xznms69+3bt6DjWbXRo0drzJgxNtv6jMIJ8rRw4UKFh4erS5cuSkpK0rVr1yRJFy5c0NixY7Vs2TKDEwIAAADAo2PTpk1as2aNwsLCVK5cOW3atElnzpxRrVq1tH379vv+vC5dumjEiBGKiIjQyJEjdebMGfXp00fh4eHmNl23lStXTrGxsWrevLk6d+6sefPmqWXLlmrUqJHatWunqKgo1ahRQydOnNB//vMftW/fXkFBQZY6dACF3MWLF81/Hj9+vAYPHqyxY8fmeWOa9TruDzN48m/ixIk5vndyclJCQoISEhJybDeZTBRO7pOttz6jcII8jR49WjExMeratavmzZtn3t64cWONHj3awGQAAAAA8OhxcXHR2rVrNWnSJF28eFHe3t6aMGGCnnnmGc2fP/++P8/R0VErVqxQv3799Pjjj8vR0VEvvPCCoqOj83x9+fLlFRsbq5CQEHXp0kVz5szRsmXLNHToUHXr1k1nzpxR+fLl1axZs1yFFwC2rVSpUjlmQmRnZys0NDTHa2gt9WCYwZN/hw4dMjqCzbL11mem7OzsbKNDoPBxdHTUrl275OPjI2dnZ6WkpMjX11cHDx6Uv7+/rl69anREAAAAAAAAGOzPT+7fDWvq/rU/zuDZunWr3n//fWbwWEhkZKQGDRokR0fHHNuvXLmijz76yOrX5CgIf2x9lpGRoejoaD333HM22fqMwgny5Ovrq6lTp6ply5Y5CiezZs3SuHHjtGvXLqMjAgAAAAAAoBA5evSoKleunGstjuzsbB07dkxeXl4GJbMednZ2uWbw5HU+mcFz/+zt7XXy5EmVK1cux/Zff/1V5cqV43zegypVqtzT60wmkw4ePPiQ0zxctOpCnnr06KF+/fpp+vTpMplMOnHihDZu3KhBgwZp2LBhRscDAAAAAABAIVOlSpU8b0yfO3dOVapU4cb0PYiLizM6gs3KqwglSSkpKSpTpowBiazPo9T6jMIJzLZv367atWvLzs5O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\n",
            "text/plain": [
              "<Figure size 2000x500 with 4 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#####################################################################\n",
        "# Calling the function PlotBarCharts() we have created\n",
        "PlotBarCharts(inpData=insurance_data, colsToPlot=['sex','children','smoker','region'])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QG-ZiRCL6YHX"
      },
      "source": [
        "##Observation from Step 9 - Visual Exploratory Data Analysis\n",
        "* Bar charts have made it possible to analyse the two data columns.\n",
        "* The category names are displayed on the X-axis of the bar charts,\n",
        "while the frequencies of each category are displayed on the Y-axis.\n",
        "* Every category in the perfect bar chart has a comparable frequency.\n",
        "* since of this, the ML/AI regression algorithm can learn since there\n",
        "are enough rows in the data for each category.\n",
        "* If there's a column in a chart that's heavily skewed, it means one\n",
        "bar stands out a lot while the others have very few instances.\n",
        "* The construction of machine learning models could not benefit\n",
        "greatly from these kinds of columns.\n",
        "* As the correlation analysis stage approaches, we may validate this\n",
        "and make a final decision on whether to accept or reject the\n",
        "column/data attribute.\n",
        "* It's important to note that \"sex\" and \"smoker\" in this dataset is biassed.\n",
        "* There is just one bar that is in control, and the other has\n",
        "relatively few rows.\n",
        "* Because there is nothing to learn from such columns, it is possible\n",
        "that they are not associated with the target variable.\n",
        "* The algorithms cannot find any rule like when the value is this then\n",
        "the target variable is that.\n",
        "* **'sex', 'children','smoker','region' **: certain categorical\n",
        "variables Two of the category variables have been chosen for\n",
        "additional examination."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nIpIXISum-je"
      },
      "source": [
        "#**Step.10 :Now Visualize distribution of all the Continuous Predictor variables in the data using histograms**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1npMvj_uoaI8"
      },
      "source": [
        "* Based on the Basic Exploratory Data Analysis, there are eleven continuous predictor variables 'age', 'bmi', 'charges'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "BR0nPtjCopG1"
      },
      "outputs": [],
      "source": [
        "# Plotting histograms of multiple columns together\n",
        "insurance_data.hist(['age', 'bmi', 'charges'], figsize=(20,12))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7BeMo1vADVQB"
      },
      "source": [
        "#**Step.11 :Feature Selection based on data distribution**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0iQh7W5ZFCRK"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.feature_selection import SelectKBest, f_classif\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "# Load the dataset\n",
        "insurance_data = pd.read_csv('Medical_insurance.csv')\n",
        "\n",
        "# Perform one-hot encoding for categorical variables\n",
        "categorical_columns = ['sex', 'smoker', 'region']  # Ensure column names are lowercase\n",
        "insurance_data_encoded = pd.get_dummies(insurance_data, columns=categorical_columns)\n",
        "\n",
        "# Define features and target variable\n",
        "X = insurance_data_encoded.drop(columns=['charges'])  # Features\n",
        "y = insurance_data_encoded['charges']  # Target variable\n",
        "\n",
        "# Use statistical tests to determine feature importance\n",
        "selector = SelectKBest(score_func=f_classif, k='all')\n",
        "selector.fit(X, y)\n",
        "feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})\n",
        "feature_scores = feature_scores.sort_values(by='Score', ascending=False)\n",
        "print(\"Feature Importance Scores:\")\n",
        "print(feature_scores)\n",
        "\n",
        "# Visualize feature importance\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(x='Score', y='Feature', data=feature_scores, palette='viridis')\n",
        "plt.title('Feature Importance Scores')\n",
        "plt.xlabel('Score')\n",
        "plt.ylabel('Feature')\n",
        "plt.show()\n",
        "\n",
        "# Select top k features based on importance scores\n",
        "k = 5  # Example: Select top 5 features\n",
        "selected_features = feature_scores['Feature'].head(k).tolist()\n",
        "print(\"Selected Features:\", selected_features)\n",
        "\n",
        "# Train a machine learning model using selected features\n",
        "X_selected = X[selected_features]\n",
        "model = RandomForestRegressor()\n",
        "model.fit(X_selected, y)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9kucwHoaqHm8"
      },
      "source": [
        "## **Step 12: Outlier Analysis**\n",
        "* Outliers are extreme values in the data which are far away from most of the values.\n",
        "* You can see them as the tails in the histogram.\n",
        "\n",
        "* Outlier must be treated one column/data attribute at a time.\n",
        "* As the treatment will be slightly different for each column\n",
        "* Why I should analyse the outliers?\n",
        "* Outliers bias the building of machine learning models.\n",
        "* As the algorithm tries to fit the extreme value, it goes away from majority of the data.\n",
        "* Outlined below are two options to treat outliers in the data.\n",
        "\n",
        "* Option-1: Delete the outlier Records. Only if there are just few rows lost.\n",
        "* Option-2: Impute the outlier values with a logical business value\n",
        "* Let us find out out the most logical value to be replaced in place of outliers by looking at the histogram.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Qzo30qtNrCiM"
      },
      "outputs": [],
      "source": [
        "#Replacing outliers for 'charges'\n",
        "# Finding nearest values to 6500 mark\n",
        "insurance_data[insurance_data['charges']<6500].sort_values(by='charges',ascending=False)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fq3cwRZqszPj"
      },
      "source": [
        "Observation: Above result shows the nearest logical value is 6496.8860, hence, replacing any value above 6500 with it."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "ZlsdkwCqs94H",
        "outputId": "1764a887-d968-4032-8b92-05eee597d420",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 158
        }
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'insurance_data' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-6f62d13798b1>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Replacing outliers with nearest possibe value\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0minsurance_data\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'charges'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0minsurance_data\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'charges'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m>\u001b[0m\u001b[0;36m6500\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m\u001b[0;36m6496.8860\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'insurance_data' is not defined"
          ]
        }
      ],
      "source": [
        "# Replacing outliers with nearest possibe value\n",
        "insurance_data['charges'][insurance_data['charges']>6500] =6496.8860"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tsVsEG4rGCPz"
      },
      "source": [
        "#**Step.13 : Removal of outliers and missing values**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jXzw4Bl0F-NB"
      },
      "outputs": [],
      "source": [
        "insurance_data.hist(['charges'], figsize=(18,5))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GsGjBjq3tyVw"
      },
      "source": [
        "## Observation from Step 13\n",
        "* The distribution has improved after the outlier treatment.\n",
        "* There is still a tail but it is thick, that means there are many values in that range, hence, it is acceptable."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RJMdJrLAt1_F"
      },
      "source": [
        "## **Step 14: Missing Values Analysis**\n",
        "\n",
        "* Missing values are treated for each column separately.\n",
        "* If a column has more than 30% data missing, then missing value treatment cannot be done.\n",
        "* That column must be rejected because too much information is missing.\n",
        "* Outlined below are some options for treating missing values in data.\n",
        "* Delete the missing value rows if there are only few records\n",
        "* Impute the missing values with MEDIAN value for continuous variables\n",
        "* Impute the missing values with MODE value for categorical variables\n",
        "* Interpolate the values based on nearby values\n",
        "* Interpolate the values based on business logic"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9dy6GrNGt_JZ"
      },
      "outputs": [],
      "source": [
        "# Finding how many missing values are there for each column\n",
        "insurance_data.isnull().sum()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0oUYcZ66ub4A"
      },
      "source": [
        "##Observations from Step 14: Missing Value Analysis\n",
        "* No missing values in this data!\n",
        "* So no removal of any data samples(rows) is needed."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZbixOMFIud_-"
      },
      "source": [
        "# **Step 15: Feature Selection (Attribute Selection)**\n",
        "\n",
        "* Now its time to finally choose the best columns(Features) which are correlated to the Target variable.\n",
        "* This can be done directly by measuring the correlation values or ANOVA analysis or Chi-Square tests.\n",
        "* However, it is always helpful to visualize the relation between the Target variable/class variable and each of the predictors(features) to get a better sense of data.\n",
        "\n",
        "* Listed below are some of the techniques used for visualizing relationship between two variables as well as measuring the strength statistically.\n",
        "\n",
        "* **Visual exploration of relationship between variables**\n",
        "* Continuous Vs Continuous ---- Scatter Plot\n",
        "* Categorical Vs Continuous---- Box Plot\n",
        "* Categorical Vs Categorical---- Grouped Bar Plots\n",
        "* Statistical measurement of relationship strength between variables\n",
        "* Continuous Vs Continuous ---- Correlation matrix\n",
        "* Categorical Vs Continuous---- ANOVA test\n",
        "* Categorical Vs Categorical--- Chi-Square test\n",
        "\n",
        "* **For this dataset, the Target variable is Continuous, hence following two scenarios will need attention**\n",
        "\n",
        "* Continuous Target Variable Vs Continuous Predictor\n",
        "* Continuous Target Variable Vs Categorical Predictor\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "A2lmJH9tu1n-"
      },
      "source": [
        "## Relationship exploration: Continuous Vs Continuous -- Scatter Charts\n",
        "* When the Target variable is continuous and the predictor is also continuous, we can visualize the relationship between the two variables using scatter plot and measure the strength of relation using a metric called pearson's correlation value."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 957
        },
        "id": "luZ-qit4uv3p",
        "outputId": "411b03ce-9b1a-4ad6-f6b4-d7ef548582ad"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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eEtAQIClZs2algcffNDy+++/G3fQZCpjxoyxNGnSxOLt7W2pX7++pV+/fpYNGzbYxvn5ospwe9t7i4WfM5I3fPhwS4MGDSze3t6Whg0bWoYPH27JzMy0jfMzRo7w3XffWdq3b2/x8fGxtGnTxrJkyZIy46WlpZa//e1vlpCQEIuPj4+lX79+loyMDIOOlpR4WCwWi5EBIRERERERkbviHDIiIiIiIiKDMCAjIiIiIiIyCAMyIiIiIiIigzAgIyIiIiIiMggDMiIiIiIiIoMwICMiIiIiIjIIAzIiIiIiIiKDMCAjIiIiIiIyCAMyIiIiIiIigzAgIyIiIiIiMggDMiIiIiIiIoMwICMiIrLDunXr0KtXL9StWxdBQUEYNGgQjh49ahvfunUrIiMj4evri27dumHNmjXw8PBAenq67TEHDhzAwIEDUbt2bYSEhODxxx/HxYsXDXg1RETkLBiQERER2aGgoABTp07Fzp07kZKSgmrVquHBBx9EaWkp8vLyMHjwYHTo0AG7d+/G66+/jhdffLHM/rm5ubj33nvRuXNn7Ny5E+vWrcP58+fxyCOPGPSKiIjIGXhYLBaL0QdBRERkNhcvXkT9+vWxf/9+bNmyBS+//DLOnDkDX19fAMDSpUvx9NNPY8+ePYiMjMSsWbOwefNmrF+/3vYzzpw5g0aNGiEjIwOtW7c26qUQEZGBmCEjIiKyw5EjRzBy5Eg0b94cfn5+aNq0KQDg1KlTyMjIQMeOHW3BGAB07969zP579+7Fxo0bUbt2bdu/Nm3aAECZ0kciInIvXkYfABERkRkMHjwYTZo0wccff4ywsDCUlpaiffv2KCoqsmv/q1evYvDgwXjzzTfLjTVo0MDRh0tERCbBgIyIiEhDTk4OMjIy8PHHHyMmJgYAsGXLFtt4REQE/vnPf+LGjRvw8fEBAPzyyy9lfkaXLl3wr3/9C02bNoWXF//8EhHRLSxZJCIi0hAQEICgoCAsWbIEmZmZ+PHHHzF16lTb+KOPPorS0lI888wzOHjwINavX4+3334bAODh4QEASEhIwKVLlzBy5Ej88ssvOHr0KNavX48nn3wSJSUlhrwuIiIyHgMyIiIiDdWqVcNXX32FXbt2oX379pgyZQreeust27ifnx++++47pKenIzIyEi+99BISExMBwDavLCwsDD///DNKSkowYMAAdOjQAZMnT0bdunVRrRr/HBMRuSt2WSQiIqoEy5cvx5NPPokrV66gRo0aRh8OERE5KRaxExEROcA//vEPNG/eHA0bNsTevXvx4osv4pFHHmEwRkREQgzIiIiIHCArKwuJiYnIyspCgwYNMGzYMPz97383+rCIiMjJsWSRiIiIiIjIIJxFTEREREREZBAGZERERERERAZhQEZERERERGQQBmREREREREQGYUBGRERERERkEAZkREREREREBmFARkREREREZBAGZERERERERAb5/0JpAR+KjR9uAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "ContinuousCols=['age','bmi']\n",
        "\n",
        "# Plotting scatter chart for each predictor vs the target variable\n",
        "for predictor in ContinuousCols:\n",
        "   insurance_data.plot.scatter(x=predictor, y='charges', figsize=(10,5), title=predictor+\" VS \"+ 'charges')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Gt9iPd_xwyt9"
      },
      "source": [
        "## Scatter charts interpretation\n",
        "* Scatter charts are great visual tools for understanding the relationship between two variables.\n",
        "\n",
        "* **Increasing Trend**: If you observe points clustering in a manner that suggests an upward slope from left to right, it indicates a positive correlation. This suggests that as one variable increases, the other tends to increase as well. It's valuable for machine learning because it indicates a potential predictive relationship.\n",
        "\n",
        "* **Decreasing Trend**: Conversely, if the points form a downward slope from left to right, it suggests a negative correlation. Here, as one variable increases, the other tends to decrease. Like an increasing trend, this is also beneficial for machine learning model building.\n",
        "No Trend: If there's no clear pattern or trend visible, it suggests little to no correlation between the variables. In this case, using this predictor as a feature in machine learning might not be very effective.\n",
        "\n",
        "\n",
        "* Based on this chart we can get a good idea about the predictor, if it will be useful or not. You confirm this by looking at the correlation value in the next step."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0bwQbn03w0QP"
      },
      "source": [
        "\n",
        "## **Step 16: Statistical Feature Selection (Continuous Vs Continuous) using Correlation value**\n",
        "\n",
        "* Pearson's correlation coefficient is a powerful metric for doing this.\n",
        "* It can simply be calculated as the covariance between two features  x and  y\n",
        "  (numerator) divided by the product of their standard deviations (denominator):\n",
        "\n",
        "![image.png](data:image/png;base64,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)\n",
        "\n",
        "* This value can be calculated only between two numeric columns\n",
        "Correlation between [-1,0) means inversely proportional, the scatter plot will show a downward trend\n",
        "* Correlation between (0,1] means directly proportional, the scatter plot will show a upward trend\n",
        "* Correlation near {0} means No relationship, the scatter plot will show no clear trend.\n",
        "* If Correlation value between two variables is > 0.5 in magnitude, it indicates good relationship the sign does not matter\n",
        "* We observe the correlations between Target variable and all other predictor variables(s) to check which columns/features/predictors are actually related to the target variable in question.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 193
        },
        "id": "9kLG2aaIx9Io",
        "outputId": "6f3d78fb-2cba-47bc-d882-97306d1c904d"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'insurance_data' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-3-f667a0a748a3>\u001b[0m in \u001b[0;36m<cell line: 5>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;31m# Creating the correlation matrix\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mCorrelationData\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0minsurance_data\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mContinuousCols\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcorr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m \u001b[0mCorrelationData\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'insurance_data' is not defined"
          ]
        }
      ],
      "source": [
        "# Calculating correlation matrix\n",
        "ContinuousCols=['age', 'charges', 'bmi',]\n",
        "\n",
        "# Creating the correlation matrix\n",
        "CorrelationData=insurance_data[ContinuousCols].corr()\n",
        "CorrelationData"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V3MMFcFkyNiS",
        "outputId": "e2734c15-cd0c-4c20-e3f2-ddb2f5107a9d"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "age        0.51576\n",
              "charges    1.00000\n",
              "Name: charges, dtype: float64"
            ]
          },
          "execution_count": 116,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Filtering only those columns where absolute correlation > 0.5 with Target Variable\n",
        "# reduce the 0.5 threshold if no variable is selected\n",
        "CorrelationData['charges'][abs(CorrelationData['charges']) > 0.5 ]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "esrEKcpfywvy"
      },
      "source": [
        "## Observations from Step 16\n",
        "* Final selected Continuous columns:\n",
        "\n",
        "* **'age'**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WPHnSkJWy9w0"
      },
      "source": [
        "# **Step 17:  Relationship exploration: Categorical Vs Continuous -- Box Plots**\n",
        "* When the target variable is Continuous and the predictor variable is Categorical we analyze the relation using Boxplots,  and\n",
        "* Measure the strength of relation using Anova test."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 450
        },
        "id": "FVQ7OSsCzHl9",
        "outputId": "2f073f5f-3c02-49c3-a916-481b81685062"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABekAAAHeCAYAAAAPaIAAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACdaUlEQVR4nOzdeXgUVdr38V+Szh4CBEjCGiMokUUQlMWFTSACMuKjMyouoIAO+46Dw7DJO8wACoggozwjjsKMiooOIhJZBwWBIApIUJBFgSRsSUgCWev9w+l+aJJAd2fp7urv57q4SKpOVd91Urm7ctfpU36GYRgCAAAAAAAAAABVzt/dAQAAAAAAAAAA4Kso0gMAAAAAAAAA4CYU6QEAAAAAAAAAcBOK9AAAAAAAAAAAuAlFegAAAAAAAAAA3IQiPQAAAAAAAAAAbkKRHgAAAAAAAAAAN6FIDwAAAAAAAACAm1CkBwAAAAAAAADATSjSAwAAANfh5+en6dOnuzsM09u8ebP8/Py0efPma7abPn26/Pz8dPbs2aoJrApYjwkAAAC+hyI9AAAA3Gb58uXy8/Oz+xcdHa2uXbvqs88+c3d45fb9999r+vTpOnbsmLtDAQAAAOChKNIDAADA7WbOnKm3335b//jHPzRp0iSdOXNGvXv31po1a9wdWrl8//33mjFjBkV6XNeUKVN06dIld4cBAAAAN7C4OwAAAACgV69euv32223fDxo0SDExMfrnP/+p+++/342ReY/CwkIVFxcrKCjI3aGYlmEYunz5skJDQyt83xaLRRYLf54BAAD4IkbSAwAAwOPUqFFDoaGhJYqWOTk5Gj9+vBo2bKjg4GA1bdpU8+bNk2EYkqRLly4pISFBCQkJdqOSz58/r7p16+rOO+9UUVGRJGngwIGKiIjQTz/9pMTERIWHh6tevXqaOXOmbX/X8s0336hXr16KjIxURESE7r33Xu3YscO2fvny5frtb38rSeratattOp/rzbf+/vvvq1mzZgoJCVGLFi300UcfaeDAgbrhhhtsbY4dOyY/Pz/NmzdPCxYsUOPGjRUcHKzvv/9ekrRx40bdc889Cg8PV40aNfTAAw/o4MGDdq9z9T6tSpsb3c/PTyNGjNCKFSvUtGlThYSEqG3bttq6dWuJ7U+ePKlnnnlGMTExCg4OVvPmzfX3v/+9RLtffvlF/fr1U3h4uKKjozV27Fjl5eVds2+udvbsWf3ud79TZGSkatWqpdGjR+vy5cu29Z07d1arVq1K3bZp06ZKTEy85v5vuOEG3X///fr88891++23KzQ0VH/7298kSRkZGRozZoztXGzSpIn++te/qri42G4f586d05NPPqnIyEjVqFFDAwYM0Lfffis/Pz8tX77c1q60fi8sLNSLL75o+/necMMNeuGFF0r0kzXObdu2qV27dgoJCdGNN96of/zjH9ftQwAAALgfQzUAAADgdpmZmTp79qwMw1B6eroWLVqk7OxsPfHEE7Y2hmHoN7/5jTZt2qRBgwapdevW+vzzzzVx4kSdPHlS8+fPV2hoqN566y3ddddd+uMf/6iXX35ZkjR8+HBlZmZq+fLlCggIsO2zqKhI9913nzp06KA5c+Zo3bp1mjZtmgoLCzVz5swy4z1w4IDuueceRUZGatKkSQoMDNTf/vY3denSRVu2bFH79u3VqVMnjRo1Sq+88opeeOEF3XLLLZJk+780n376qR555BG1bNlSs2fP1oULFzRo0CDVr1+/1PZvvvmmLl++rGeffVbBwcGKiorSF198oV69eunGG2/U9OnTdenSJS1atEh33XWX9uzZU2ph3hFbtmzRu+++q1GjRik4OFhLlizRfffdp507d6pFixaSpLS0NHXo0MFW1K9Tp44+++wzDRo0SFlZWRozZoykX2+m3HvvvTpx4oRGjRqlevXq6e2339bGjRudiul3v/udbrjhBs2ePVs7duzQK6+8ogsXLtiK008++aSGDBmi/fv322KUpF27dumHH37QlClTrvsahw4d0mOPPabnnntOQ4YMUdOmTZWbm6vOnTvr5MmTeu6559SoUSN99dVXmjx5sk6fPq0FCxZIkoqLi9W3b1/t3LlTQ4cOVUJCgj7++GMNGDDAoeMbPHiw3nrrLT388MMaP368vv76a82ePVsHDx7URx99ZNf28OHDevjhhzVo0CANGDBAf//73zVw4EC1bdtWzZs3d7BHAQAA4BYGAAAA4CZvvvmmIanEv+DgYGP58uV2bVevXm1IMmbNmmW3/OGHHzb8/PyMw4cP25ZNnjzZ8Pf3N7Zu3Wq8//77hiRjwYIFdtsNGDDAkGSMHDnStqy4uNjo06ePERQUZJw5c8a2XJIxbdo02/f9+vUzgoKCjCNHjtiWnTp1yqhW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t2+3fVrjShkZGbYbTImJiSooKNAbb7xhW19cXFwi7tJYf1YLFiywW/7yyy9Lkvr06eNq+AAAAPAwjKQHAACA23322We2EfHp6elauXKlfvzxR/3hD3+wFbz79u2rrl276o9//KOOHTumVq1aaf369fr44481ZswYNW7cWJI0a9Ys7d27Vxs2bFC1atV06623aurUqZoyZYoefvhhu0J1SEiI1q1bpwEDBqh9+/b67LPP9Omnn+qFF14oc75x62skJSXp7rvv1rBhw2SxWPS3v/1NeXl5mjNnjq1d69atFRAQoL/+9a/KzMxUcHCwunXrpujo6FL3++c//1kPPPCA7rrrLj399NO6cOGCXn31VbVo0cKucH8tc+fOVa9evdSxY0cNGjRIly5d0qJFi1S9enVNnz7d1u7RRx/V888/rwcffFCjRo1Sbm6uXnvtNd18882lzn/fokULJSYmatSoUQoODtaSJUskSTNmzLC1+ctf/qJNmzapffv2GjJkiJo1a6bz589rz549+uKLL3T+/HlJ0pAhQ/Tqq6/qqaeeUnJysurWrau33367xNzt13P06FH95je/0X333aft27frnXfeUf/+/dWqVSu7drfddptatGih999/X7fccovatGnj1OtcbeLEifrkk090//33a+DAgWrbtq1ycnK0b98+rVq1SseOHVPt2rXVr18/tWvXTuPHj9fhw4eVkJCgTz75xNYP1/qkRatWrTRgwAC9/vrrysjIUOfOnbVz50699dZb6tevn7p27VquYwAAAIAHMQAAAAA3efPNNw1Jdv9CQkKM1q1bG6+99ppRXFxs1/7ixYvG2LFjjXr16hmBgYHGTTfdZMydO9fWLjk52bBYLMbIkSPttissLDTuuOMOo169esaFCxcMwzCMAQMGGOHh4caRI0eMnj17GmFhYUZMTIwxbdo0o6ioyG57Sca0adPslu3Zs8dITEw0IiIijLCwMKNr167GV199VeIY33jjDePGG280AgICDEnGpk2brtkn//rXv4yEhAQjODjYaNGihfHJJ58YDz30kJGQkGBrc/ToUUOSMXfu3FL38cUXXxh33XWXERoaakRGRhp9+/Y1vv/++xLt1q9fb7Ro0cIICgoymjZtarzzzjvGtGnTjKv/TJBkDB8+3HjnnXeMm266yQgODjZuu+22Uo8lLS3NGD58uNGw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d34DFSUvL88ICAgwPvroI7vlTz31lPGb3/zGPUFVIEkljs2b/fjjj4YkY9++fe4OpUIUFhYa//znP42goCDjwIED7g7HZU899ZQxZswYwzB+fY8YPXq0ewMqh2nTphlhYWFG3bp1jfj4eKN///7G8ePH3R0WfFDnzp2NyMhIY/r06cYPP/xgvPXWW4afn5+xfv16o6ioyGjdurVx9913G7t37zZ27NhhtG3b1ujcubO7w4aXyM3NNapXr2689957tmVpaWmGxWIxNm7caGzdutWIjIw0li9fbhw5csRYv369ccMNNxjTp083DMMw3n//fSMyMtJYu3atcfz4cePrr782Xn/9dXcdDrxIeXPbtGnTjPDwcOO+++4z9uzZY3z77bdGRkaG0bFjR2PIkCHG6dOnjdOnTxuFhYW2mkj79u2NzZs3GwcOHDDuuece484777Tt73rnumEYxvz5842NGzcaR48eNTZs2GA0bdrUGDp0qG19nz59jB49ehjfffedceTIEePf//63sWXLFqOwsND44IMPDEnGoUOHjNOnTxsZGRle32dt2rQx5s6daxiGYezdu9eIiooygoKCjIsXLxqGYRiDBw82Hn/8ccMwDCM/P9+45ZZbjGeeecb47rvvjO+//97o37+/0bRpUyMvL88oKCgwqlevbkyYMME4fPiw8f333xvLly83jh8/buTm5hrjx483mjdvbosxNzfXqb47deqUYbFYjJdfftk4evSo8d133xmLFy82Ll68+P/bu/e4nM//D+CvSoe781FKuR0jVHLIWshW0yPTso21vpYYNsSY5fQwsixsFg3bd5PvMr4zZmOHGCUKtaU5NIfKyvE30YZQo1Hv3x89+nzdOlJ2k9fz8fB4+Fyfz+e635/L5eq63933dcnSpUvF3NxcvvzyS8nNzZUZM2aIvr6+nDhxQkT+l1O7cuWKUt+hQ4cEgJw6dUrKysokLi5OzM3Nlfiq2kCtVou1tbV89NFH8ttvv8miRYtEV1dXcnNzG9QuIiIpKSmybt06ycnJkePHj8uYMWPE3t5erl27JiJ1j8OXLl0SJycniY6OVmJ71Nts+fLl0q1bN6XeHj16iK2trfz73/8WEZF9+/aJvr6+lJaWikj9OaWIiAjp0aOHZGVlyalTpyQ5OVm+//57ERHZv3+/AJCdO3dKYWGhXLp0qcHtdycm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OzsJCwuTP//8U7k/OjpaWrVqJTo6Ovc1H38Y22zLli0CQNnUU0RkypQpAkByc3M1XqewsFBGjhwptra2YmhoKO3bt5dx48bJ1atX5cKFCzJ06FBl7qVWq2XevHlSXl4uIpUbgL744otiaWl5X+8Bjh8/LgEBAWJnZyeGhobi4uIiK1asEJHKjVjnz58vrVu3Fn19fY3Nb6vs27dP3NzcxMjISPr37y+bNm2q1hbjx48XGxsbASBRUVEiUnN+xMPDQzlfX7uIiBw8eFB69+4tRkZG0qlTJ9m0aZNGvfWNwz/99JO4u7uLoaGh3Eu6+GFus/LycrGyspK+ffsqZVX9c9asWRr31pdTWrBggbi6uopKpRJra2sJDg6WkydPKvfHx8eLs7Oz6Orqiq+vb4Pb7046Incs/ENEREREREREVAs/Pz9069YNy5cv13YoREREzQaT9ERERERERERUpytXriA1NRXDhg3D8ePH0blzZ22HRERE1GxwTXoiIiIiIiIiqpOnpyeuXLmC9957jwl6IiKiJsZP0hMRERERERERERERaYmutgMgIiIiIiIiIiIiInpcMUlPRERERERERERERKQlTNITEREREREREREREWkJk/RERERERERERERERFrCJD0RERE9dk6fPg0dHR0cPny41mvWrFkDS0tL5Xj+/Pno0aNHnfWOGjUKQ4cObZIYiYjo4cNxnoioeWjbti3i4uK0HQaRooW2AyAiIiJ6GIWEhGDw4MHaDoOIiIiIiJpYVlYWTExMtB0GkYJJeiIiIqIaqFQqqFSqJq3z77//hoGBQZPWSUREjy7+XCAiujdNNW7a2dk1QTRETYfL3RA1ga+//hpubm5QqVSwsbGBv78/SktLAQCrV6+Gq6srjIyM0KVLF3z88cfKfa+++irc3d1RVlYGoPKHjaenJ0aOHKmV5yAiam4qKirw/vvvo2PHjjA0NESbNm0QExOjnD958iSeeuopGBsbw8PDAz/99JNy7u7lbu5WXl6OadOmwdLSEjY2NpgxYwZEROOagQMHYtKkSZg6dSpsbW0REBAAADh69CgCAwNhamoKe3t7hIWF4c8//9S474033sCMGTNgbW2NVq1aYf78+U3TKEREzUht8/CqZWkWLlwIe3t7WFpaIjo6Grdv38b06dNhbW0NJycnJCQkaNR35MgRPP3000p9r732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\n",
            "text/plain": [
              "<Figure size 1800x500 with 4 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Box plots for continuous Target Variable \"charges\" and Categorical predictors\n",
        "CategoricalColsList=['sex', 'children', 'smoker', 'region']\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(CategoricalColsList), figsize=(18,5))\n",
        "\n",
        "# Creating box plots for each continuous predictor against the Target Variable \"MEDV\"\n",
        "for PredictorCol , i in zip(CategoricalColsList, range(len(CategoricalColsList))):\n",
        "    insurance_data.boxplot(column='charges', by=PredictorCol, figsize=(5,5), vert=True, ax=PlotCanvas[i])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tZpwOat-zrAC"
      },
      "source": [
        "##Observations from Step 17: Box-Plots interpretation\n",
        "\n",
        "\n",
        "* These plots gives an idea about the data distribution of continuous predictor in the Y-axis for each of the category in the X-Axis.\n",
        "\n",
        "* If the distribution looks similar for each category(Boxes are in the same line), that means the the continuous variable has NO effect on the target variable. Hence, the variables are not correlated to each other.\n",
        "\n",
        "* On the other hand if the distribution is different for each category(the boxes are not in same line!). It hints that these variables might be correlated with MEDV.\n",
        "\n",
        "* For this datadata, both the categorical predictors looks correlated with the Target variable.\n",
        "\n",
        "We confirm this by looking at the results of ANOVA test below"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5fbf_JdTzwKb"
      },
      "source": [
        "## **Step 18: Statistical Feature Selection (Categorical Vs Continuous) using ANOVA test**\n",
        "\n",
        "* Analysis of variance(ANOVA) is performed to check if there is any relationship between the given continuous and categorical variable\n",
        "\n",
        "* Assumption(H0) Null Hypothesis: There is NO relation between the given variables (i.e.\n",
        "* The average(mean) values of the numeric Target variable is same for all the groups in the categorical Predictor variable)\n",
        "* ANOVA Test result: Probability of H0 (Null Hypothesis being true"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3ANFZfpLz2W0"
      },
      "outputs": [],
      "source": [
        "# Defining a function to find the statistical relationship with all the categorical variables\n",
        "def FunctionAnova(inpData, TargetVariable, CategoricalPredictorList):\n",
        "    from scipy.stats import f_oneway\n",
        "\n",
        "    # Creating an empty list of final selected predictors\n",
        "    SelectedPredictors=[]\n",
        "\n",
        "    print('##### ANOVA Results ##### \\n')\n",
        "    for predictor in CategoricalPredictorList:\n",
        "        CategoryGroupLists=inpData.groupby(predictor)[TargetVariable].apply(list)\n",
        "        AnovaResults = f_oneway(*CategoryGroupLists)\n",
        "\n",
        "        # If the ANOVA P-Value is <0.05, that means we reject H0\n",
        "        if (AnovaResults[1] < 0.05):\n",
        "            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])\n",
        "            SelectedPredictors.append(predictor)\n",
        "        else:\n",
        "            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])\n",
        "\n",
        "    return(SelectedPredictors)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W6uysQqw0Mm0",
        "outputId": "39fd7ef9-5281-4f51-9a7a-dfddcc2bc8b4"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "##### ANOVA Results ##### \n",
            "\n",
            "sex is NOT correlated with charges | P-Value: 0.12753368412626134\n",
            "children is correlated with charges | P-Value: 3.0118185046581406e-07\n",
            "smoker is correlated with charges | P-Value: 2.474527727919898e-205\n",
            "region is correlated with charges | P-Value: 0.000734846336935172\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "['children', 'smoker', 'region']"
            ]
          },
          "execution_count": 107,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "#Calling the function to check which categorical variables are correlated with target\n",
        "CategoricalPredictorList=['sex', 'children', 'smoker', 'region']\n",
        "FunctionAnova(inpData=insurance_data,\n",
        "              TargetVariable='charges',\n",
        "              CategoricalPredictorList=CategoricalPredictorList)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bLA2AeXV00zc"
      },
      "source": [
        "##Observations from Step 18\n",
        "* The results of ANOVA confirm our visual analysis using box plots above.\n",
        "\n",
        "* All categorical variables are correlated with the Target variable except 'sex'.\n",
        "* This is something we can guess by looking at the box plots!\n",
        "\n",
        "* Final selected Categorical columns:\n",
        "\n",
        " **'children', 'smoker', 'region'**\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Q-nTz6fA1F18"
      },
      "source": [
        "## Selecting final Predictors/Features for building Machine Learning/AI model.\n",
        "* Based on the extensive tests with exploratory data analysis, we can select the final features/predictors/columns for machine learning model building as:\n",
        "* **'age', 'children', 'smoker', 'region'**\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "YloxzTCf1jCM",
        "outputId": "d12ca125-3eb3-43a7-9819-32fc212908d1"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"DataForML\",\n  \"rows\": 2302,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13,\n        \"min\": 18,\n        \"max\": 64,\n        \"num_unique_values\": 47,\n        \"samples\": [\n          34,\n          49,\n          53\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"children\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 5,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0,\n          1,\n          4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"smoker\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"no\",\n          \"yes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"region\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"southeast\",\n          \"northeast\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "DataForML"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-391ce7d0-8b89-4d7a-a619-7dd09d0b47ae\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>children</th>\n",
              "      <th>smoker</th>\n",
              "      <th>region</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>19</td>\n",
              "      <td>0</td>\n",
              "      <td>yes</td>\n",
              "      <td>southwest</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>18</td>\n",
              "      <td>1</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>28</td>\n",
              "      <td>3</td>\n",
              "      <td>no</td>\n",
              "      <td>southeast</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>33</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>northwest</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>32</td>\n",
              "      <td>0</td>\n",
              "      <td>no</td>\n",
              "      <td>northwest</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-391ce7d0-8b89-4d7a-a619-7dd09d0b47ae')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-391ce7d0-8b89-4d7a-a619-7dd09d0b47ae button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-391ce7d0-8b89-4d7a-a619-7dd09d0b47ae');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-fe4780ed-294d-465c-beba-8e0a672d576b\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-fe4780ed-294d-465c-beba-8e0a672d576b')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-fe4780ed-294d-465c-beba-8e0a672d576b button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "   age  children smoker     region\n",
              "0   19         0    yes  southwest\n",
              "1   18         1     no  southeast\n",
              "2   28         3     no  southeast\n",
              "3   33         0     no  northwest\n",
              "4   32         0     no  northwest"
            ]
          },
          "execution_count": 108,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "SelectedColumns=['age', 'children', 'smoker', 'region']\n",
        "\n",
        "# Selecting final columns\n",
        "DataForML=insurance_data[SelectedColumns]\n",
        "DataForML.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "l5FWBIEP1yHE"
      },
      "outputs": [],
      "source": [
        "# Saving this final data subset for reference during deployment\n",
        "DataForML.to_pickle('DataForML.pkl')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GIGJ5ejd1z4a"
      },
      "source": [
        "## **Step 19: Data Pre-processing for Machine Learning Model Building or Model Development**\n",
        "* List of steps that needs to be performed on predictor variables before data can be used for machine learning\n",
        "\n",
        "* Converting each Ordinal Categorical columns to numeric\n",
        "* Converting Binary nominal Categorical columns to numeric using 1/0 mapping\n",
        "* Converting all other nominal categorical columns to numeric using pd.get_dummies()\n",
        "* Data Transformation (Optional): Standardization/Normalization/log/sqrt. Important if you are using distance based algorithms like KNN, or Neural Networks\n",
        "* Converting the ordinal variable to numeric - In this data there is no Ordinal categorical variable.\n",
        "* Converting the binary nominal variable to numeric using 1/0 mapping: There is no binary nominal variable in string format in this data\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nEQ5esz32Mc5"
      },
      "source": [
        "## Converting the nominal variable to numeric using get_dummies()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "jLl1rNMS2J3D",
        "outputId": "8baae243-77a7-4365-b99f-69a82cbc0f0b"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"DataForML_Numeric\",\n  \"rows\": 2302,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13,\n        \"min\": 18,\n        \"max\": 64,\n        \"num_unique_values\": 47,\n        \"samples\": [\n          34,\n          49,\n          53\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"children\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 5,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0,\n          1,\n          4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"smoker_no\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          true,\n          false\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"smoker_yes\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          false,\n          true\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"region_northeast\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          true,\n          false\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"region_northwest\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          true,\n          false\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"region_southeast\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          true,\n          false\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"region_southwest\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          false,\n          true\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"charges\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 5565.408042829472,\n        \"min\": 1121.8739,\n        \"max\": 24227.33724,\n        \"num_unique_values\": 1112,\n        \"samples\": [\n          9861.025,\n          10579.711\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "DataForML_Numeric"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-cd456cfe-aff6-44d1-b8b5-7c286488822e\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>children</th>\n",
              "      <th>smoker_no</th>\n",
              "      <th>smoker_yes</th>\n",
              "      <th>region_northeast</th>\n",
              "      <th>region_northwest</th>\n",
              "      <th>region_southeast</th>\n",
              "      <th>region_southwest</th>\n",
              "      <th>charges</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>19</td>\n",
              "      <td>0</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>16884.92400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>18</td>\n",
              "      <td>1</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>1725.55230</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>28</td>\n",
              "      <td>3</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>4449.46200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>33</td>\n",
              "      <td>0</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>21984.47061</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>32</td>\n",
              "      <td>0</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>3866.85520</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-cd456cfe-aff6-44d1-b8b5-7c286488822e')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-cd456cfe-aff6-44d1-b8b5-7c286488822e button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-cd456cfe-aff6-44d1-b8b5-7c286488822e');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-bddddbbe-761d-48fb-9aa6-a7c2943c90f5\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-bddddbbe-761d-48fb-9aa6-a7c2943c90f5')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-bddddbbe-761d-48fb-9aa6-a7c2943c90f5 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "   age  children  smoker_no  smoker_yes  region_northeast  region_northwest  \\\n",
              "0   19         0      False        True             False             False   \n",
              "1   18         1       True       False             False             False   \n",
              "2   28         3       True       False             False             False   \n",
              "3   33         0       True       False             False              True   \n",
              "4   32         0       True       False             False              True   \n",
              "\n",
              "   region_southeast  region_southwest      charges  \n",
              "0             False              True  16884.92400  \n",
              "1              True             False   1725.55230  \n",
              "2              True             False   4449.46200  \n",
              "3             False             False  21984.47061  \n",
              "4             False             False   3866.85520  "
            ]
          },
          "execution_count": 110,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Treating all the nominal variables at once using dummy variables\n",
        "DataForML_Numeric=pd.get_dummies(DataForML)\n",
        "\n",
        "# Adding Target Variable to the data\n",
        "DataForML_Numeric['charges']=insurance_data['charges']\n",
        "\n",
        "# Printing sample rows\n",
        "DataForML_Numeric.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1eVywKEn2fHi"
      },
      "source": [
        "## **Step 20: Machine Learning Model Development:**\n",
        "* Splitting the data into Training and Testing sample\n",
        "* We dont use the full data for creating the model (training data).\n",
        "* Some data is randomly selected and kept aside for checking how good the model is.\n",
        "* This is known as Testing Data and the remaining data is called Training data on which the model is built.\n",
        "* Typically 70% of data is used as Training data and the rest 30% is used as Tesing data."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FfMTCU6T2mWH"
      },
      "outputs": [],
      "source": [
        "# Printing all the column names for our reference\n",
        "DataForML_Numeric.columns"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "s-K5moKu3Bf5"
      },
      "outputs": [],
      "source": [
        "#Separate Target Variable and Predictor Variables\n",
        "TargetVariable='charges'\n",
        "Predictors=['age', 'children', ]\n",
        "\n",
        "X=DataForML_Numeric[Predictors].values\n",
        "y=DataForML_Numeric[TargetVariable].values\n",
        "\n",
        "# Split the data into training and testing set\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cHejUDYk5AA9"
      },
      "source": [
        "## **Step 21: Standardization/Normalization of data**\n",
        "* we can choose not to run this step if we want to compare the resultant accuracy of this transformation with the accuracy of raw data (Optional Step)\n",
        "\n",
        "* However, if we are using KNN or Neural Networks, then this step becomes necessary."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7kYf8Uq55Kjk"
      },
      "outputs": [],
      "source": [
        "### Sandardization of data ###\n",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
        "# Choose either standardization or Normalization\n",
        "# On this data Min Max Normalization produced better results\n",
        "\n",
        "# Choose between standardization and MinMAx normalization\n",
        "#PredictorScaler=StandardScaler()\n",
        "PredictorScaler=MinMaxScaler()\n",
        "\n",
        "# Storing the fit object for later reference\n",
        "PredictorScalerFit=PredictorScaler.fit(X)\n",
        "\n",
        "# Generating the standardized values of X\n",
        "X=PredictorScalerFit.transform(X)\n",
        "\n",
        "# Split the data into training and testing set\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CPHC5xm15Q8k",
        "outputId": "774984bf-4e04-43c6-8e6e-bc1c23ed3e0e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(1611, 2)\n",
            "(1611,)\n",
            "(691, 2)\n",
            "(691,)\n"
          ]
        }
      ],
      "source": [
        "# Sanity check for the sampled data\n",
        "print(X_train.shape)\n",
        "print(y_train.shape)\n",
        "print(X_test.shape)\n",
        "print(y_test.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gAb84TlM5VNX"
      },
      "source": [
        "## **Step 22: Multiple Linear Regression Algorithm For ML/AI model building**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ptcGMoS05afn",
        "outputId": "566da76c-4c66-4e6a-f859-e4c4efc5c82d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "LinearRegression()\n",
            "R2 Value: 0.2849193445761986\n",
            "\n",
            "##### Model Validation and Accuracy Calculations ##########\n",
            "        age  children     charges  Predictedcharges\n",
            "0  0.695652       0.6  11085.5868           12113.0\n",
            "1  0.086957       0.0   2150.4690            4696.0\n",
            "2  1.000000       0.0  14394.5579           13315.0\n",
            "3  0.717391       0.2   9877.6077           11204.0\n",
            "4  0.782609       0.0  10923.9332           11263.0\n",
            "Mean Accuracy on test data: 50.23625821926316\n",
            "Median Accuracy on test data: 54.94237920473189\n",
            "\n",
            "Accuracy values for 10-fold Cross Validation:\n",
            " [44.54868764 48.74282426 45.4959398  50.9435754  46.69113993 44.3760031\n",
            " 48.9241458  45.08229209 51.0586865  46.66928472]\n",
            "\n",
            "Final Average Accuracy of the model: 47.25\n"
          ]
        }
      ],
      "source": [
        "#Multiple Linear Regression\n",
        "from sklearn.linear_model import LinearRegression\n",
        "RegModel = LinearRegression()\n",
        "\n",
        "# Printing all the parameters of Linear regression\n",
        "print(RegModel)\n",
        "\n",
        "# Creating the model on Training Data\n",
        "LREG=RegModel.fit(X_train,y_train)\n",
        "prediction=LREG.predict(X_test)\n",
        "\n",
        "from sklearn import metrics\n",
        "# Measuring Goodness of fit in Training data\n",
        "print('R2 Value:',metrics.r2_score(y_train, LREG.predict(X_train)))\n",
        "\n",
        "###########################################################################\n",
        "print('\\n##### Model Validation and Accuracy Calculations ##########')\n",
        "\n",
        "# Printing some sample values of prediction\n",
        "TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)\n",
        "TestingDataResults[TargetVariable]=y_test\n",
        "TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)\n",
        "\n",
        "# Printing sample prediction values\n",
        "print(TestingDataResults.head())\n",
        "\n",
        "# Calculating the error for each row\n",
        "TestingDataResults['APE']=100 * ((abs(\n",
        "  TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])\n",
        "\n",
        "MAPE=np.mean(TestingDataResults['APE'])\n",
        "MedianMAPE=np.median(TestingDataResults['APE'])\n",
        "\n",
        "Accuracy =100 - MAPE\n",
        "MedianAccuracy=100- MedianMAPE\n",
        "print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier\n",
        "print('Median Accuracy on test data:', MedianAccuracy)\n",
        "\n",
        "# Defining a custom function to calculate accuracy\n",
        "# Make sure there are no zeros in the Target variable if you are using MAPE\n",
        "def Accuracy_Score(orig,pred):\n",
        "    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))\n",
        "    #print('#'*70,'Accuracy:', 100-MAPE)\n",
        "    return(100-MAPE)\n",
        "\n",
        "# Custom Scoring MAPE calculation\n",
        "from sklearn.metrics import make_scorer\n",
        "custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)\n",
        "\n",
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zpKF6MBbGyRV"
      },
      "source": [
        "# **Decision Regressor Tree**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "i8KV0zvyHMGj"
      },
      "outputs": [],
      "source": [
        "# Decision Trees (Multiple if-else statements!)\n",
        "from sklearn.tree import DecisionTreeRegressor\n",
        "RegModel = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse')\n",
        "# Good Range of Max_depth = 2 to 20\n",
        "\n",
        "# Printing all the parameters of Decision Tree\n",
        "print(RegModel)\n",
        "\n",
        "# Creating the model on Training Data\n",
        "DT=RegModel.fit(X_train,y_train)\n",
        "prediction=DT.predict(X_test)\n",
        "\n",
        "from sklearn import metrics\n",
        "# Measuring Goodness of fit in Training data\n",
        "print('R2 Value:',metrics.r2_score(y_train, DT.predict(X_train)))\n",
        "\n",
        "# Plotting the feature importance for Top 10 most important columns\n",
        "%matplotlib inline\n",
        "feature_importances = pd.Series(DT.feature_importances_, index=Predictors)\n",
        "feature_importances.nlargest(10).plot(kind='barh')\n",
        "\n",
        "###########################################################################\n",
        "print('\\n##### Model Validation and Accuracy Calculations ##########')\n",
        "\n",
        "# Printing some sample values of prediction\n",
        "TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)\n",
        "TestingDataResults[TargetVariable]=y_test\n",
        "TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)\n",
        "\n",
        "# Printing sample prediction values\n",
        "print(TestingDataResults.head())\n",
        "\n",
        "# Calculating the error for each row\n",
        "TestingDataResults['APE']=100 * ((abs(\n",
        "    TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])\n",
        "\n",
        "MAPE=np.mean(TestingDataResults['APE'])\n",
        "MedianMAPE=np.median(TestingDataResults['APE'])\n",
        "\n",
        "Accuracy =100 - MAPE\n",
        "MedianAccuracy=100- MedianMAPE\n",
        "print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier\n",
        "print('Median Accuracy on test data:', MedianAccuracy)\n",
        "\n",
        "# Defining a custom function to calculate accuracy\n",
        "# Make sure there are no zeros in the Target variable if you are using MAPE\n",
        "def Accuracy_Score(orig,pred):\n",
        "    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))\n",
        "    #print('#'*70,'Accuracy:', 100-MAPE)\n",
        "    return(100-MAPE)\n",
        "\n",
        "# Custom Scoring MAPE calculation\n",
        "from sklearn.metrics import make_scorer\n",
        "custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)\n",
        "\n",
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ASia_r3IHrOe"
      },
      "source": [
        "# Plotting/Visualising the Decision Tree"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "heW_yvmcHw_Y"
      },
      "outputs": [],
      "source": [
        "\n",
        "# Load libraries\n",
        "from IPython.display import Image\n",
        "from sklearn import tree\n",
        "import pydotplus\n",
        "\n",
        "# Create DOT data\n",
        "dot_data = tree.export_graphviz(RegModel, out_file=None,\n",
        "                                feature_names=Predictors, class_names=TargetVariable)\n",
        "\n",
        "# printing the rules\n",
        "#print(dot_data)\n",
        "\n",
        "# Draw graph\n",
        "graph = pydotplus.graph_from_dot_data(dot_data)\n",
        "\n",
        "# Show graph\n",
        "Image(graph.create_png(), width=1600,height=1000)\n",
        "# Double click on the graph to zoom in"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VKNwnCnLIoHC"
      },
      "source": [
        "# Random Forest Regressor"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RQu2xJIOIpVq"
      },
      "outputs": [],
      "source": [
        "# Random Forest (Bagging of multiple Decision Trees)\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "RegModel = RandomForestRegressor(max_depth=4, n_estimators=400,criterion='friedman_mse')\n",
        "# Good range for max_depth: 2-10 and n_estimators: 100-1000\n",
        "\n",
        "# Printing all the parameters of Random Forest\n",
        "print(RegModel)\n",
        "\n",
        "# Creating the model on Training Data\n",
        "RF=RegModel.fit(X_train,y_train)\n",
        "prediction=RF.predict(X_test)\n",
        "\n",
        "from sklearn import metrics\n",
        "# Measuring Goodness of fit in Training data\n",
        "print('R2 Value:',metrics.r2_score(y_train, RF.predict(X_train)))\n",
        "\n",
        "# Plotting the feature importance for Top 10 most important columns\n",
        "%matplotlib inline\n",
        "feature_importances = pd.Series(RF.feature_importances_, index=Predictors)\n",
        "feature_importances.nlargest(10).plot(kind='barh')\n",
        "\n",
        "###########################################################################\n",
        "print('\\n##### Model Validation and Accuracy Calculations ##########')\n",
        "\n",
        "# Printing some sample values of prediction\n",
        "TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)\n",
        "TestingDataResults[TargetVariable]=y_test\n",
        "TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)\n",
        "\n",
        "# Printing sample prediction values\n",
        "print(TestingDataResults.head())\n",
        "\n",
        "# Calculating the error for each row\n",
        "TestingDataResults['APE']=100 * ((abs(\n",
        "    TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])\n",
        "\n",
        "MAPE=np.mean(TestingDataResults['APE'])\n",
        "MedianMAPE=np.median(TestingDataResults['APE'])\n",
        "\n",
        "Accuracy =100 - MAPE\n",
        "MedianAccuracy=100- MedianMAPE\n",
        "print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier\n",
        "print('Median Accuracy on test data:', MedianAccuracy)\n",
        "\n",
        "\n",
        "# Defining a custom function to calculate accuracy\n",
        "# Make sure there are no zeros in the Target variable if you are using MAPE\n",
        "def Accuracy_Score(orig,pred):\n",
        "    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))\n",
        "    #print('#'*70,'Accuracy:', 100-MAPE)\n",
        "    return(100-MAPE)\n",
        "\n",
        "# Custom Scoring MAPE calculation\n",
        "from sklearn.metrics import make_scorer\n",
        "custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)\n",
        "\n",
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gJD0w_j4I_Id"
      },
      "source": [
        "# Plotting One of the Decision Tree in Random Forest Regressor"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LEscswT_I__M"
      },
      "outputs": [],
      "source": [
        "# Plotting a single Decision Tree from Random Forest\n",
        "# Load libraries\n",
        "from IPython.display import Image\n",
        "from sklearn import tree\n",
        "import pydotplus\n",
        "\n",
        "# Create DOT data for the 6th Decision Tree in Random Forest\n",
        "dot_data = tree.export_graphviz(RegModel.estimators_[5] , out_file=None, feature_names=Predictors, class_names=TargetVariable)\n",
        "\n",
        "# Draw graph\n",
        "graph = pydotplus.graph_from_dot_data(dot_data)\n",
        "\n",
        "# Show graph\n",
        "Image(graph.create_png(), width=1600,height=1000)\n",
        "# Double click on the graph to zoom in"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NB7BQ8uV5wEL"
      },
      "source": [
        "\n",
        "## **Step 23: AdaBoost Algorithm For ML/AI model building**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lqk4mL2V51sH"
      },
      "outputs": [],
      "source": [
        "# Adaboost (Boosting of multiple Decision Trees)\n",
        "from sklearn.ensemble import AdaBoostRegressor\n",
        "from sklearn.tree import DecisionTreeRegressor\n",
        "\n",
        "# Choosing Decision Tree with 6 level as the weak learner\n",
        "DTR=DecisionTreeRegressor(max_depth=3)\n",
        "RegModel = AdaBoostRegressor(n_estimators=100, base_estimator=DTR ,learning_rate=0.04)\n",
        "\n",
        "# Printing all the parameters of Adaboost\n",
        "print(RegModel)\n",
        "\n",
        "# Creating the model on Training Data\n",
        "AB=RegModel.fit(X_train,y_train)\n",
        "prediction=AB.predict(X_test)\n",
        "\n",
        "from sklearn import metrics\n",
        "# Measuring Goodness of fit in Training data\n",
        "print('R2 Value:',metrics.r2_score(y_train, AB.predict(X_train)))\n",
        "\n",
        "# Plotting the feature importance for Top 10 most important columns\n",
        "%matplotlib inline\n",
        "feature_importances = pd.Series(AB.feature_importances_, index=Predictors)\n",
        "feature_importances.nlargest(10).plot(kind='barh')\n",
        "\n",
        "###########################################################################\n",
        "print('\\n##### Model Validation and Accuracy Calculations ##########')\n",
        "\n",
        "# Printing some sample values of prediction\n",
        "TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)\n",
        "TestingDataResults[TargetVariable]=y_test\n",
        "TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)\n",
        "\n",
        "# Printing sample prediction values\n",
        "print(TestingDataResults.head())\n",
        "\n",
        "# Calculating the error for each row\n",
        "TestingDataResults['APE']=100 * ((abs(\n",
        "  TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])\n",
        "\n",
        "MAPE=np.mean(TestingDataResults['APE'])\n",
        "MedianMAPE=np.median(TestingDataResults['APE'])\n",
        "\n",
        "Accuracy =100 - MAPE\n",
        "MedianAccuracy=100- MedianMAPE\n",
        "print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier\n",
        "print('Median Accuracy on test data:', MedianAccuracy)\n",
        "\n",
        "\n",
        "# Defining a custom function to calculate accuracy\n",
        "# Make sure there are no zeros in the Target variable if you are using MAPE\n",
        "def Accuracy_Score(orig,pred):\n",
        "    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))\n",
        "    #print('#'*70,'Accuracy:', 100-MAPE)\n",
        "    return(100-MAPE)\n",
        "\n",
        "# Custom Scoring MAPE calculation\n",
        "from sklearn.metrics import make_scorer\n",
        "custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)\n",
        "\n",
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_wiCfGrwJeNZ"
      },
      "source": [
        "# XGBoost Regressor"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "dOv6urKV6LxG",
        "outputId": "d5356f18-2d16-4651-85a1-fb634910badf"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "XGBRegressor(base_score=None, booster='gbtree', callbacks=None,\n",
            "             colsample_bylevel=None, colsample_bynode=None,\n",
            "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
            "             enable_categorical=False, eval_metric=None, feature_types=None,\n",
            "             gamma=None, grow_policy=None, importance_type=None,\n",
            "             interaction_constraints=None, learning_rate=0.1, max_bin=None,\n",
            "             max_cat_threshold=None, max_cat_to_onehot=None,\n",
            "             max_delta_step=None, max_depth=2, max_leaves=None,\n",
            "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
            "             multi_strategy=None, n_estimators=1000, n_jobs=None,\n",
            "             num_parallel_tree=None, objective='reg:linear', ...)\n",
            "R2 Value: 0.35489892488397834\n",
            "\n",
            "##### Model Validation and Accuracy Calculations ##########\n",
            "        age  children     charges  Predictedcharges\n",
            "0  0.695652       0.6  11085.5868           11995.0\n",
            "1  0.086957       0.0   2150.4690            2047.0\n",
            "2  1.000000       0.0  14394.5579           14269.0\n",
            "3  0.717391       0.2   9877.6077           10404.0\n",
            "4  0.782609       0.0  10923.9332           10607.0\n",
            "Mean Accuracy on test data: 51.200862368850316\n",
            "Median Accuracy on test data: 67.57015125489849\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [06:54:26] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n",
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [06:54:26] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n",
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [06:54:27] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Accuracy values for 10-fold Cross Validation:\n",
            " [46.31239572 49.89737935 45.81600496 54.96003049 48.6743459  46.15201256\n",
            " 50.0973787  45.53912362 54.91048814 48.95330287]\n",
            "\n",
            "Final Average Accuracy of the model: 49.13\n"
          ]
        },
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkUAAAGdCAYAAAAc+wceAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAZqUlEQVR4nO3de2zVhf3/8VdLpUWl4GVi0Q4CDt0UBkokePl6mZcpY+rMmLoRdZvOiJnKDMx5QbwgIosuU3dhoHNzkrk5s4sBHeofMOcFrZGLKCjRqbCgs4BmRejn98d+NKsX7Cm90O7xSE5iz/mc0/e7h9Knn57SsqIoigAA/I8r7+wBAAB2BKIIACCiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIklR09gBdRWNjY95444307t07ZWVlnT0OANACRVFkw4YN6d+/f8rLt30uSBS10BtvvJHa2trOHgMAaIXXXnst++677zaPEUUt1Lt37yT/+aBWV1d38jQAQEusX78+tbW1TV/Ht0UUtdDWb5lVV1eLIgDoYlry0hcvtAYAiCgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJIkFZ09QFdz0JT5Ka/cubPH6DZWTx/T2SMAQBJnigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBI0oZRtHr16pSVlaWuru5jj7nrrrvSt2/fprevueaaDB8+fJuPe8455+TUU09tkxkBAD5Oh54p+trXvpYXX3yxI98lAECLVHTkO+vVq1d69erVpo+5adOm9OzZs00fEwD431PymaLGxsbMmDEj++23XyorK/PpT386N9xwQ9PtL7/8co455pjsvPPO+fznP5/HH3+86bYPfvvsg7Zs2ZKJEyemb9++2WOPPTJp0qQURdHsmKOPPjoXXXRRLrnkkuy555458cQTkyRLlizJSSedlF133TX9+vXL+PHjs27dumb3++53v5tJkyZl9913z957751rrrmm1PUBgG6q5Ci6/PLLM3369Fx11VVZtmxZfvOb36Rfv35Nt19xxRW57LLLUldXlyFDhuTMM8/M5s2bW/TYP/zhD3PXXXdlzpw5WbhwYd5+++384Q9/+NBxv/zlL9OzZ88sWrQoP/3pT/POO+/k2GOPzYgRI/L0009n3rx5Wbt2bcaNG/eh++2yyy554oknMmPGjFx77bV5+OGHP3KWhoaGrF+/vtkFAOi+Svr22YYNG/KjH/0ot912W84+++wkyeDBg3PEEUdk9erVSZLLLrssY8aMSZJMnTo1Bx54YFauXJkDDjjgEx//1ltvzeWXX56vfOUrSZKf/vSnmT9//oeO+8xnPpMZM2Y0vX399ddnxIgRmTZtWtN1c+bMSW1tbV588cUMGTIkSTJs2LBMmTKl6TFuu+22LFiwIMcff/yH3seNN96YqVOntuTDAgB0AyWdKVq+fHkaGhryhS984WOPGTZsWNN/19TUJEn++c9/fuJj19fX580338yoUaOarquoqMjIkSM/dOwhhxzS7O3nnnsujz76aHbdddemy9YIW7Vq1UfOtnW+j5vt8ssvT319fdPltdde+8QdAICuq6QzRS15kfROO+3U9N9lZWVJ/vM6pLa0yy67NHt748aNGTt2bG666aYPHbs1zD4429b5Pm62ysrKVFZWtsG0AEBXUNKZos985jPp1atXFixY0OaD9OnTJzU1NXniiSeartu8eXMWL178ifc9+OCDs3Tp0gwcODD77bdfs8sHAwoA4KOUFEVVVVWZPHlyJk2alLvvvjurVq3K3//+98yePbtNhrn44oszffr0PPDAA3nhhRdy4YUX5p133vnE+02YMCFvv/12zjzzzDz11FNZtWpV5s+fn3PPPTdbtmxpk9kAgO6t5H+n6KqrrkpFRUWuvvrqvPHGG6mpqckFF1zQJsN873vfy5tvvpmzzz475eXl+eY3v5nTTjst9fX127xf//79s2jRokyePDknnHBCGhoaMmDAgHzxi19MebnfZAIAfLKy4oP/EBAfaf369enTp09qL/ltyit37uxxuo3V08d09ggAdGNbv37X19enurp6m8c6jQIAEFEEAJBEFAEAJBFFAABJRBEAQBJRBACQRBQBACQRRQAASUQRAEASUQQAkEQUAQAkEUUAAElEEQBAElEEAJBEFAEAJBFFAABJRBEAQBJRBACQRBQBACQRRQAASUQRAEASUQQAkEQUAQAkEUUAAElEEQBAElEEAJBEFAEAJBFFAABJRBEAQBJRBACQRBQBACQRRQAASUQRAEASUQQAkEQUAQAkEUUAAElEEQBAElEEAJBEFAEAJEkqOnuArmbJ1BNTXV3d2WMAAG3MmSIAgIgiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASJJUdPYAXc1BU+anvHLnzh4DALqV1dPHdPYIzhQBACSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBI0oWiaN68eTniiCPSt2/f7LHHHvnSl76UVatWNd3+t7/9LcOHD09VVVVGjhyZBx54IGVlZamrq2s6ZsmSJTnppJOy6667pl+/fhk/fnzWrVvXCdsAADuaLhNF7777biZOnJinn346CxYsSHl5eU477bQ0NjZm/fr1GTt2bIYOHZpnnnkm1113XSZPntzs/u+8806OPfbYjBgxIk8//XTmzZuXtWvXZty4cR/5/hoaGrJ+/fpmFwCg+6ro7AFa6vTTT2/29pw5c/KpT30qy5Yty8KFC1NWVpZZs2alqqoqn/vc5/L666/nvPPOazr+tttuy4gRIzJt2rRmj1FbW5sXX3wxQ4YMafb4N954Y6ZOndq+SwEAO4wuc6bopZdeyplnnplBgwaluro6AwcOTJK8+uqrWbFiRYYNG5aqqqqm4w899NBm93/uuefy6KOPZtddd226HHDAAUnS7NtwW11++eWpr69vurz22mvttxwA0Om6zJmisWPHZsCAAZk1a1b69++fxsbGHHTQQdm0aVOL7r9x48aMHTs2N91004duq6mp+dB1lZWVqays3O65AYCuoUtE0VtvvZUVK1Zk1qxZOfLII5MkCxcubLp9//33z69//es0NDQ0hcxTTz3V7DEOPvjg/P73v8/AgQNTUdEl1gYAOlCX+PbZbrvtlj322CM///nPs3LlyjzyyCOZOHFi0+1nnXVWGhsbc/7552f58uWZP39+Zs6cmSQpKytLkkyYMCFvv/12zjzzzDz11FNZtWpV5s+fn3PPPTdbtmzplL0AgB1Hl4ii8vLyzJ07N4sXL85BBx2USy+9NDfffHPT7dXV1fnTn/6Uurq6DB8+PFdccUWuvvrqJGl6nVH//v2zaNGibNmyJSeccEKGDh2aSy65JH379k15eZf4MAAA7aisKIqis4doD/fcc0/OPffc1NfXp1evXtv9eOvXr0+fPn1Se8lvU165cxtMCABstXr6mHZ53K1fv+vr61NdXb3NY7vNi2vuvvvuDBo0KPvss0+ee+65TJ48OePGjWuTIAIAur9uE0Vr1qzJ1VdfnTVr1qSmpiZf/epXc8MNN3T2WABAF9FtomjSpEmZNGlSZ48BAHRRXmEMABBRBACQRBQBACQRRQAASUQRAEASUQQAkEQUAQAkEUUAAElEEQBAElEEAJBEFAEAJBFFAABJRBEAQBJRBACQRBQBACQRRQAASUQRAEASUQQAkEQUAQAkEUUAAElEEQBAElEEAJBEFAEAJBFFAABJRBEAQBJRBACQRBQBACQRRQAASUQRAEASUQQAkEQUAQAkEUUAAElEEQBAElEEAJBEFAEAJBFFAABJkorOHqCrWTL1xFRXV3f2GABAG3OmCAAgoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSiCIAgCSiCAAgiSgCAEgiigAAkogiAIAkoggAIIkoAgBIIooAAJKIIgCAJKIIACCJKAIASCKKAACSJBWdPUBXURRFkmT9+vWdPAkA0FJbv25v/Tq+LaKohd56660kSW1tbSdPAgCUasOGDenTp882jxFFLbT77rsnSV599dVP/KB2RevXr09tbW1ee+21VFdXd/Y4bc5+XVd33i2xX1fXnffrLrsVRZENGzakf//+n3isKGqh8vL/vPyqT58+XfoPxyeprq62XxfWnffrzrsl9uvquvN+3WG3lp7M8EJrAICIIgCAJKKoxSorKzNlypRUVlZ29ijtwn5dW3ferzvvltivq+vO+3Xn3T5OWdGSn1EDAOjmnCkCAIgoAgBIIooAAJKIIgCAJKKomdtvvz0DBw5MVVVVRo0alSeffHKbx99333054IADUlVVlaFDh+bBBx/soElbp5T9li5dmtNPPz0DBw5MWVlZbr311o4btJVK2W/WrFk58sgjs9tuu2W33XbLcccd94nPd2crZb/7778/I0eOTN++fbPLLrtk+PDh+dWvftWB05am1M+9rebOnZuysrKceuqp7Tvgdiplv7vuuitlZWXNLlVVVR04belKff7eeeedTJgwITU1NamsrMyQIUN26L8/S9nv6KOP/tDzV1ZWljFjxnTgxC1X6nN36623Zv/990+vXr1SW1ubSy+9NP/+9787aNoOUFAURVHMnTu36NmzZzFnzpxi6dKlxXnnnVf07du3WLt27Ucev2jRoqJHjx7FjBkzimXLlhVXXnllsdNOOxXPP/98B0/eMqXu9+STTxaXXXZZce+99xZ77713ccstt3TswCUqdb+zzjqruP3224tnn322WL58eXHOOecUffr0Kf7xj3908OQtU+p+jz76aHH//fcXy5YtK1auXFnceuutRY8ePYp58+Z18OSfrNTdtnrllVeKffbZpzjyyCOLU045pWOGbYVS97vzzjuL6urq4s0332y6rFmzpoOnbrlS92toaChGjhxZnHzyycXChQuLV155pXjssceKurq6Dp68ZUrd76233mr23C1ZsqTo0aNHceedd3bs4C1Q6m733HNPUVlZWdxzzz3FK6+8UsyfP7+oqakpLr300g6evP2Iov/v0EMPLSZMmND09pYtW4r+/fsXN95440ceP27cuGLMmDHNrhs1alTxne98p13nbK1S9/tvAwYM2OGjaHv2K4qi2Lx5c9G7d+/il7/8ZXuNuF22d7+iKIoRI0YUV155ZXuMt11as9vmzZuLww47rPjFL35RnH322Tt0FJW635133ln06dOng6bbfqXu95Of/KQYNGhQsWnTpo4acbts7+feLbfcUvTu3bvYuHFje43YaqXuNmHChOLYY49tdt3EiROLww8/vF3n7Ei+fZZk06ZNWbx4cY477rim68rLy3Pcccfl8ccf/8j7PP74482OT5ITTzzxY4/vTK3Zrytpi/3ee++9vP/++02/+HdHsr37FUWRBQsWZMWKFfm///u/9hy1ZK3d7dprr81ee+2Vb33rWx0xZqu1dr+NGzdmwIABqa2tzSmnnJKlS5d2xLgla81+f/zjHzN69OhMmDAh/fr1y0EHHZRp06Zly5YtHTV2i7XF3y2zZ8/OGWeckV122aW9xmyV1ux22GGHZfHixU3fYnv55Zfz4IMP5uSTT+6QmTuCXwibZN26ddmyZUv69evX7Pp+/frlhRde+Mj7rFmz5iOPX7NmTbvN2Vqt2a8raYv9Jk+enP79+38odHcErd2vvr4+++yzTxoaGtKjR4/ccccdOf7449t73JK0ZreFCxdm9uzZqaur64AJt09r9tt///0zZ86cDBs2LPX19Zk5c2YOO+ywLF26NPvuu29HjN1irdnv5ZdfziOPPJKvf/3refDBB7Ny5cpceOGFef/99zNlypSOGLvFtvfvlieffDJLlizJ7Nmz22vEVmvNbmeddVbWrVuXI444IkVRZPPmzbngggvygx/8oCNG7hCiiP9506dPz9y5c/PYY4/t8C9oLUXv3r1TV1eXjRs3ZsGCBZk4cWIGDRqUo48+urNHa7UNGzZk/PjxmTVrVvbcc8/OHqddjB49OqNHj256+7DDDstnP/vZ/OxnP8t1113XiZO1jcbGxuy11175+c9/nh49euSQQw7J66+/nptvvnmHi6LtNXv27AwdOjSHHnpoZ4/SJh577LFMmzYtd9xxR0aNGpWVK1fm4osvznXXXZerrrqqs8drE6IoyZ577pkePXpk7dq1za5fu3Zt9t5774+8z957713S8Z2pNft1Jduz38yZMzN9+vT89a9/zbBhw9pzzFZr7X7l5eXZb7/9kiTDhw/P8uXLc+ONN+5QUVTqbqtWrcrq1aszduzYpusaGxuTJBUVFVmxYkUGDx7cvkOXoC0+93baaaeMGDEiK1eubI8Rt0tr9qupqclOO+2UHj16NF332c9+NmvWrMmmTZvSs2fPdp25FNvz/L377ruZO3durr322vYcsdVas9tVV12V8ePH59vf/naSZOjQoXn33Xdz/vnn54orrkh5edd/RU7X36AN9OzZM4ccckgWLFjQdF1jY2MWLFjQ7P/Y/tvo0aObHZ8kDz/88Mce35las19X0tr9ZsyYkeuuuy7z5s3LyJEjO2LUVmmr56+xsTENDQ3tMWKrlbrbAQcckOeffz51dXVNly9/+cs55phjUldXl9ra2o4c/xO1xXO3ZcuWPP/886mpqWmvMVutNfsdfvjhWblyZVPMJsmLL76YmpqaHSqIku17/u677740NDTkG9/4RnuP2Sqt2e299977UPhsjduiu/wa1U5+ofcOY+7cuUVlZWVx1113FcuWLSvOP//8om/fvk0/Cjt+/Pji+9//ftPxixYtKioqKoqZM2cWy5cvL6ZMmbLD/0h+Kfs1NDQUzz77bPHss88WNTU1xWWXXVY8++yzxUsvvdRZK2xTqftNnz696NmzZ/G73/2u2Y/PbtiwobNW2KZS95s2bVrx0EMPFatWrSqWLVtWzJw5s6ioqChmzZrVWSt8rFJ3+6Ad/afPSt1v6tSpxfz584tVq1YVixcvLs4444yiqqqqWLp0aWetsE2l7vfqq68WvXv3Li666KJixYoVxZ///Odir732Kq6//vrOWmGbWvvn84gjjii+9rWvdfS4JSl1tylTphS9e/cu7r333uLll18uHnrooWLw4MHFuHHjOmuFNieK/suPf/zj4tOf/nTRs2fP4tBDDy3+/ve/N9121FFHFWeffXaz43/7298WQ4YMKXr27FkceOCBxV/+8pcOnrg0pez3yiuvFEk+dDnqqKM6fvAWKmW/AQMGfOR+U6ZM6fjBW6iU/a644opiv/32K6qqqorddtutGD16dDF37txOmLplSv3c+287ehQVRWn7XXLJJU3H9uvXrzj55JOLZ555phOmbrlSn7+//e1vxahRo4rKyspi0KBBxQ033FBs3ry5g6duuVL3e+GFF4okxUMPPdTBk5aulN3ef//94pprrikGDx5cVFVVFbW1tcWFF15Y/Otf/+r4wdtJWVF0l3NeAACt5zVFAAARRQAASUQRAEASUQQAkEQUAQAkEUUAAElEEQBAElEEAJBEFAEAJBFFAABJRBEAQBJRBACQJPl/BT77slc2//4AAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "##XGBoost\n",
        "# Xtreme Gradient Boosting (XGBoost)\n",
        "from xgboost import XGBRegressor\n",
        "RegModel=XGBRegressor(max_depth=2,\n",
        "                      learning_rate=0.1,\n",
        "                      n_estimators=1000,\n",
        "                      objective='reg:linear',\n",
        "                      booster='gbtree')\n",
        "\n",
        "# Printing all the parameters of XGBoost\n",
        "print(RegModel)\n",
        "\n",
        "# Creating the model on Training Data\n",
        "XGB=RegModel.fit(X_train,y_train)\n",
        "prediction=XGB.predict(X_test)\n",
        "\n",
        "from sklearn import metrics\n",
        "# Measuring Goodness of fit in Training data\n",
        "print('R2 Value:',metrics.r2_score(y_train, XGB.predict(X_train)))\n",
        "\n",
        "# Plotting the feature importance for Top 10 most important columns\n",
        "%matplotlib inline\n",
        "feature_importances = pd.Series(XGB.feature_importances_, index=Predictors)\n",
        "feature_importances.nlargest(10).plot(kind='barh')\n",
        "###########################################################################\n",
        "print('\\n##### Model Validation and Accuracy Calculations ##########')\n",
        "\n",
        "# Printing some sample values of prediction\n",
        "TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)\n",
        "TestingDataResults[TargetVariable]=y_test\n",
        "TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)\n",
        "\n",
        "# Printing sample prediction values\n",
        "print(TestingDataResults.head())\n",
        "\n",
        "# Calculating the error for each row\n",
        "TestingDataResults['APE']=100 * ((abs(\n",
        " TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])\n",
        "\n",
        "\n",
        "MAPE=np.mean(TestingDataResults['APE'])\n",
        "MedianMAPE=np.median(TestingDataResults['APE'])\n",
        "\n",
        "Accuracy =100 - MAPE\n",
        "MedianAccuracy=100- MedianMAPE\n",
        "print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier\n",
        "print('Median Accuracy on test data:', MedianAccuracy)\n",
        "\n",
        "\n",
        "# Defining a custom function to calculate accuracy\n",
        "# Make sure there are no zeros in the Target variable if you are using MAPE\n",
        "def Accuracy_Score(orig,pred):\n",
        "    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))\n",
        "    #print('#'*70,'Accuracy:', 100-MAPE)\n",
        "    return(100-MAPE)\n",
        "\n",
        "# Custom Scoring MAPE calculation\n",
        "from sklearn.metrics import make_scorer\n",
        "custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)\n",
        "\n",
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_s8xvS3DJovg"
      },
      "source": [
        "#Plotting a single Decision tree out of XGBoost"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nozKwFQa6dIa"
      },
      "outputs": [],
      "source": [
        "#Plotting a single Decision tree out of XGBoost\n",
        "from xgboost import plot_tree\n",
        "import matplotlib.pyplot as plt\n",
        "fig, ax = plt.subplots(figsize=(20, 8))\n",
        "plot_tree(XGB, num_trees=10, ax=ax)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "02LZlZfGJuSs"
      },
      "source": [
        "# K-Nearest Neighbor(KNN)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z17TVLGI6h_f",
        "outputId": "ddda55a8-477b-4cc7-d3f2-0788b27e170e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "KNeighborsRegressor(n_neighbors=3)\n",
            "R2 Value: 0.23904939605863407\n",
            "\n",
            "##### Model Validation and Accuracy Calculations ##########\n",
            "        age  children     charges  Predictedcharges\n",
            "0  0.695652       0.6  11085.5868           16700.0\n",
            "1  0.086957       0.0   2150.4690            2119.0\n",
            "2  1.000000       0.0  14394.5579           14209.0\n",
            "3  0.717391       0.2   9877.6077            9736.0\n",
            "4  0.782609       0.0  10923.9332           10335.0\n",
            "Mean Accuracy on test data: 38.69953482392084\n",
            "Median Accuracy on test data: 91.11958498228414\n",
            "\n",
            "Accuracy values for 10-fold Cross Validation:\n",
            " [32.65906373 27.87133498 29.94995915 56.08125755 40.4005985  41.05984199\n",
            " 41.1653788  34.65055172 40.58555178 33.86725944]\n",
            "\n",
            "Final Average Accuracy of the model: 37.83\n"
          ]
        }
      ],
      "source": [
        "#kNN\n",
        "# K-Nearest Neighbor(KNN)\n",
        "from sklearn.neighbors import KNeighborsRegressor\n",
        "RegModel = KNeighborsRegressor(n_neighbors=3)\n",
        "\n",
        "# Printing all the parameters of KNN\n",
        "print(RegModel)\n",
        "\n",
        "# Creating the model on Training Data\n",
        "KNN=RegModel.fit(X_train,y_train)\n",
        "prediction=KNN.predict(X_test)\n",
        "\n",
        "from sklearn import metrics\n",
        "# Measuring Goodness of fit in Training data\n",
        "print('R2 Value:',metrics.r2_score(y_train, KNN.predict(X_train)))\n",
        "\n",
        "# Plotting the feature importance for Top 10 most important columns\n",
        "# The variable importance chart is not available for KNN\n",
        "\n",
        "###########################################################################\n",
        "print('\\n##### Model Validation and Accuracy Calculations ##########')\n",
        "\n",
        "# Printing some sample values of prediction\n",
        "TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)\n",
        "TestingDataResults[TargetVariable]=y_test\n",
        "TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)\n",
        "\n",
        "# Printing sample prediction values\n",
        "print(TestingDataResults.head())\n",
        "\n",
        "# Calculating the error for each row\n",
        "TestingDataResults['APE']=100 * ((abs(\n",
        "  TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])\n",
        "\n",
        "MAPE=np.mean(TestingDataResults['APE'])\n",
        "MedianMAPE=np.median(TestingDataResults['APE'])\n",
        "\n",
        "Accuracy =100 - MAPE\n",
        "MedianAccuracy=100- MedianMAPE\n",
        "print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier\n",
        "print('Median Accuracy on test data:', MedianAccuracy)\n",
        "\n",
        "# Defining a custom function to calculate accuracy\n",
        "# Make sure there are no zeros in the Target variable if you are using MAPE\n",
        "def Accuracy_Score(orig,pred):\n",
        "    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))\n",
        "    #print('#'*70,'Accuracy:', 100-MAPE)\n",
        "    return(100-MAPE)\n",
        "\n",
        "# Custom Scoring MAPE calculation\n",
        "from sklearn.metrics import make_scorer\n",
        "custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)\n",
        "\n",
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6TEO8OwbJ5Hy"
      },
      "source": [
        "# Support Vector Machine (SVM) Regressor"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I6aXnRSIJ8Nr",
        "outputId": "55d474c0-28d7-4711-b672-eff9039e52e7"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "SVR(C=50, gamma=0.01)\n",
            "R2 Value: 0.016365864281955855\n",
            "\n",
            "##### Model Validation and Accuracy Calculations ##########\n",
            "        age     charges  children  Predictedcharges\n",
            "0  0.695652  11085.5868       0.6            7836.0\n",
            "1  0.086957   2150.4690       0.0            7498.0\n",
            "2  1.000000  14394.5579       0.0            7957.0\n",
            "3  0.717391   9877.6077       0.2            7813.0\n",
            "4  0.782609  10923.9332       0.0            7840.0\n",
            "Mean Accuracy on test data: 21.10821322308172\n",
            "Median Accuracy on test data: 57.22231239377077\n",
            "\n",
            "Accuracy values for 10-fold Cross Validation:\n",
            " [10.73283093 15.96997037 12.15011671 24.78302563 16.91885244 10.57613849\n",
            " 15.99814255 11.9095669  24.648104   17.11144048]\n",
            "\n",
            "Final Average Accuracy of the model: 16.08\n"
          ]
        }
      ],
      "source": [
        "# Support Vector Machines(SVM)\n",
        "from sklearn import svm\n",
        "RegModel = svm.SVR(C=50, kernel='rbf', gamma=0.01)\n",
        "\n",
        "# Printing all the parameters\n",
        "print(RegModel)\n",
        "\n",
        "# Creating the model on Training Data\n",
        "SVM=RegModel.fit(X_train,y_train)\n",
        "prediction=SVM.predict(X_test)\n",
        "\n",
        "from sklearn import metrics\n",
        "# Measuring Goodness of fit in Training data\n",
        "print('R2 Value:',metrics.r2_score(y_train, SVM.predict(X_train)))\n",
        "\n",
        "# Plotting the feature importance for Top 10 most important columns\n",
        "# The built in attribute SVM.coef_ works only for linear kernel\n",
        "%matplotlib inline\n",
        "#feature_importances = pd.Series(SVM.coef_[0], index=Predictors)\n",
        "#feature_importances.nlargest(10).plot(kind='barh')\n",
        "\n",
        "###########################################################################\n",
        "print('\\n##### Model Validation and Accuracy Calculations ##########')\n",
        "\n",
        "# Printing some sample values of prediction\n",
        "TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)\n",
        "TestingDataResults[TargetVariable]=y_test\n",
        "TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)\n",
        "\n",
        "# Printing sample prediction values\n",
        "print(TestingDataResults.head())\n",
        "\n",
        "# Calculating the error for each row\n",
        "TestingDataResults['APE']=100 * ((abs(\n",
        "  TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])\n",
        "\n",
        "MAPE=np.mean(TestingDataResults['APE'])\n",
        "MedianMAPE=np.median(TestingDataResults['APE'])\n",
        "\n",
        "Accuracy =100 - MAPE\n",
        "MedianAccuracy=100- MedianMAPE\n",
        "print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier\n",
        "print('Median Accuracy on test data:', MedianAccuracy)\n",
        "\n",
        "# Defining a custom function to calculate accuracy\n",
        "# Make sure there are no zeros in the Target variable if you are using MAPE\n",
        "def Accuracy_Score(orig,pred):\n",
        "    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))\n",
        "    #print('#'*70,'Accuracy:', 100-MAPE)\n",
        "    return(100-MAPE)\n",
        "\n",
        "# Custom Scoring MAPE calculation\n",
        "from sklearn.metrics import make_scorer\n",
        "custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)\n",
        "\n",
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cazmItUe64PA"
      },
      "source": [
        "## **Step 24: Model Deployment**\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EMYP2r6QKH35"
      },
      "source": [
        "* Deployment of the Model - Based on the above trials we select that algorithm which produces the best average accuracy.\n",
        "\n",
        "* In this case, multiple algorithms have produced similar kind of average accuracy. Hence, we can choose any one of them.\n",
        "\n",
        "* I am choosing XGboost as the final model it has the highest accuracy!\n",
        "\n",
        "* In order to deploy the model we follow steps outlined next.\n",
        "\n",
        "* Train/Build the model again using 100% data available\n",
        "\n",
        "* Save the model as a serialized file which can be stored anywhere.\n",
        "\n",
        "* Create a python function which gets integrated with front-end Viewer(GUI/ Website etc.) to take all the inputs and returns the prediction\n",
        "\n",
        "* Choosing only the most important variables\n",
        "\n",
        "* Its beneficial to keep lesser number of predictors for the model while deploying it in production.\n",
        "\n",
        "* The lesser predictors you keep, the better it is, because the model will be less dependent on predictor columns/features, hence, more stable.\n",
        "\n",
        "* This is important specially when the data is high dimensional(too many predictor columns/features).\n",
        "\n",
        "* For this dataset, the most important predictor variables are **'age', 'children'**. As these are consistently on top of the variable importance chart for every algorithm. Hence choosing these as final set of predictor variables will result in better house price prediction platform/system."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FS_WwWU5KMgz",
        "outputId": "345b7767-7129-4360-bab1-d0f8e86a78b0"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(2302, 2)\n",
            "(2302,)\n"
          ]
        }
      ],
      "source": [
        "# Separate Target Variable and Predictor Variables\n",
        "TargetVariable='charges'\n",
        "\n",
        "# Selecting the final set of predictors for the deployment\n",
        "# Based on the variable importance charts of multiple algorithms above\n",
        "Predictors=['age', 'children']\n",
        "\n",
        "X=DataForML_Numeric[Predictors].values\n",
        "y=DataForML_Numeric[TargetVariable].values\n",
        "\n",
        "### Sandardization of data ###\n",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
        "# Choose either standardization or Normalization\n",
        "# On this data Min Max Normalization produced better results\n",
        "\n",
        "# Choose between standardization and MinMAx normalization\n",
        "#PredictorScaler=StandardScaler()\n",
        "PredictorScaler=MinMaxScaler()\n",
        "\n",
        "# Storing the fit object for later reference\n",
        "PredictorScalerFit=PredictorScaler.fit(X)\n",
        "\n",
        "# Generating the standardized values of X\n",
        "X=PredictorScalerFit.transform(X)\n",
        "\n",
        "print(X.shape)\n",
        "print(y.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UjmyHql4Mp_a"
      },
      "source": [
        "# Cross validating the final model accuracy with less predictors"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zhNiRB8mL-7N",
        "outputId": "b4187fbb-fbf6-4b44-d835-6f18d44edaf5"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [07:00:30] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n",
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [07:00:31] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n",
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [07:00:35] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n",
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [07:00:39] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n",
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [07:00:40] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Accuracy values for 10-fold Cross Validation:\n",
            " [50.31695962 52.55103169 47.0681668  62.17920519 48.34483186 50.11474896\n",
            " 52.40642512 47.02429815 62.17901588 48.49645051]\n",
            "\n",
            "Final Average Accuracy of the model: 52.07\n"
          ]
        }
      ],
      "source": [
        "# Importing cross validation function from sklearn\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "# choose from different tunable hyper parameters\n",
        "from xgboost import XGBRegressor\n",
        "RegModel=XGBRegressor(max_depth=6,\n",
        "                      learning_rate=0.7,\n",
        "                      n_estimators=1000,\n",
        "                      objective='reg:linear',\n",
        "                      booster='gbtree')\n",
        "\n",
        "# Running 10-Fold Cross validation on a given algorithm\n",
        "# Passing full data X and y because the K-fold will split the data and automatically choose train/test\n",
        "Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)\n",
        "print('\\nAccuracy values for 10-fold Cross Validation:\\n',Accuracy_Values)\n",
        "print('\\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ITTxkJbBM92j"
      },
      "source": [
        "# **Step 25: Retraining the final model using 100% data**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vs99N-gGNc4X",
        "outputId": "df1d2a3f-b8dc-49ac-8cf1-5bcc6e66b081"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [07:01:55] WARNING: /workspace/src/objective/regression_obj.cu:209: reg:linear is now deprecated in favor of reg:squarederror.\n",
            "  warnings.warn(smsg, UserWarning)\n"
          ]
        }
      ],
      "source": [
        "# Training the model on 100% Data available\n",
        "Final_XGB_Model=RegModel.fit(X,y)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iLslwjILNu3-"
      },
      "source": [
        "# **Step 26: Save the model as a serialized file which can be stored anywhere**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yxDRHbN1N3h4",
        "outputId": "913f7b82-80dd-4543-92ad-6e65d1b602dd"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "pickle file of Predictive Model is saved at Location: /content/drive/MyDrive/Capstone Project- Medical Insurance Price Prediction\n"
          ]
        }
      ],
      "source": [
        "import pickle\n",
        "import os\n",
        "\n",
        "# Saving the Python objects as serialized files can be done using pickle library\n",
        "# Here let us save the Final model\n",
        "with open('Final_XGB_Model.pkl', 'wb') as fileWriteStream:\n",
        "    pickle.dump(Final_XGB_Model, fileWriteStream)\n",
        "    # Don't forget to close the filestream!\n",
        "    fileWriteStream.close()\n",
        "\n",
        "print('pickle file of Predictive Model is saved at Location:',os.getcwd())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "K1Mt0WucN8xt"
      },
      "source": [
        "# **Step 27: Create a python function**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DCNeckH5OC2T"
      },
      "outputs": [],
      "source": [
        "from re import IGNORECASE\n",
        "# This Function can be called from any from any front end tool/website\n",
        "\n",
        "def FunctionPredictResult(InputData):\n",
        "    import pandas as pd\n",
        "    Num_Inputs=InputData.shape[0]\n",
        "\n",
        "    # Making sure the input data has same columns as it was used for training the model\n",
        "    # Also, if standardization/normalization was done, then same must be done for new input\n",
        "\n",
        "    # Appending the new data with the Training data\n",
        "    DataForML=pd.read_pickle('DataForML.pkl')\n",
        "    #InputData=InputData.append(DataForML, ignore_index=True)\n",
        "    InputData = pd.concat([InputData, DataForML], ignore_index=True)\n",
        "\n",
        "    # Generating dummy variables for rest of the nominal variables\n",
        "    InputData=pd.get_dummies(InputData)\n",
        "\n",
        "    # Maintaining the same order of columns as it was during the model training\n",
        "    Predictors=['age', 'children']\n",
        "\n",
        "    # Generating the input values to the model\n",
        "    X=InputData[Predictors].values[0:Num_Inputs]\n",
        "\n",
        "    # Generating the standardized values of X since it was done while model training also\n",
        "    X=PredictorScalerFit.transform(X)\n",
        "\n",
        "    # Loading the Function from pickle file\n",
        "    import pickle\n",
        "    with open('Final_XGB_Model.pkl', 'rb') as fileReadStream:\n",
        "        PredictionModel=pickle.load(fileReadStream)\n",
        "        # Don't forget to close the filestream!\n",
        "        fileReadStream.close()\n",
        "\n",
        "    # Genrating Predictions\n",
        "    Prediction=PredictionModel.predict(X)\n",
        "    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])\n",
        "    return(PredictionResult)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iFZn9yLFOj04"
      },
      "source": [
        "# **Step 28: Calling the function for some new data**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1EDTOlqoSS0g",
        "outputId": "307b8113-8dd4-48d0-81da-bfecde49dcc1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "        Prediction\n",
            "0     14321.889648\n",
            "1     16971.015625\n",
            "2     14321.889648\n",
            "3     16971.015625\n",
            "4     16971.015625\n",
            "...            ...\n",
            "2299  16971.015625\n",
            "2300  14321.889648\n",
            "2301  16971.015625\n",
            "2302  14321.889648\n",
            "2303  14321.889648\n",
            "\n",
            "[2304 rows x 1 columns]\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Define the function for prediction\n",
        "def FunctionPredictResult(InputData):\n",
        "    # Load necessary libraries and models\n",
        "    with open('Final_XGB_Model.pkl', 'rb') as fileReadStream:\n",
        "        PredictionModel = pickle.load(fileReadStream)\n",
        "\n",
        "    # Load the data used for model training\n",
        "    DataForML = pd.read_pickle('DataForML.pkl')\n",
        "\n",
        "    # Combine the new input data with the training data\n",
        "    InputData = pd.concat([InputData, DataForML], ignore_index=True)\n",
        "\n",
        "    # Ensure that the input data has the same columns as it was used for training\n",
        "    Predictors = ['age', 'children']\n",
        "\n",
        "    # Extract the relevant features and generate dummy variables if necessary\n",
        "    InputData = InputData[Predictors]\n",
        "\n",
        "    # If there are nominal variables requiring dummy encoding, you can apply pd.get_dummies here\n",
        "\n",
        "    # Assuming PredictorScalerFit is defined elsewhere and used for standardization\n",
        "    # X = PredictorScalerFit.transform(InputData)\n",
        "\n",
        "    # Generate predictions\n",
        "    Predictions = PredictionModel.predict(InputData)\n",
        "\n",
        "    # Create a DataFrame to store the predictions\n",
        "    PredictionResult = pd.DataFrame(Predictions, columns=['Prediction'])\n",
        "\n",
        "    return PredictionResult\n",
        "\n",
        "# Define the new sample data\n",
        "NewSampleData = pd.DataFrame(data=[[21, 0], [28, 3]], columns=['age', 'children'])\n",
        "\n",
        "# Call the function to predict on the new data\n",
        "prediction_result = FunctionPredictResult(NewSampleData)\n",
        "\n",
        "# Print the prediction result\n",
        "print(prediction_result)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FgvigPWqSmZt"
      },
      "source": [
        "# Conclusion\n",
        "* The Function FunctionPredictResult() can be used to produce the predictions for one or more new cases at a time.\n",
        "* Hence, it can be scheduled using a batch job or cron job to run every night and generate predictions for all the house price tasks  in the platform /system."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "acbG7BsES8Sh"
      },
      "source": [
        "# Deploying a predictive model as an API\n",
        "* Django and flask are two popular ways to deploy predictive models as a web service\n",
        "* You can call your predictive models using a URL from any front end like tableau, java or angular js\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n-mzOtoHTBSu"
      },
      "source": [
        "# Deploying the model with few parameters\n",
        "# Function for predictions API"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vQNTNSr1T4ZD",
        "outputId": "586e1353-03cd-48e2-b0b0-557e0252dd18"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{\"Prediction\":{\"0\":14321.8896484375,\"1\":14321.8896484375,\"2\":16971.015625,\"3\":16971.015625,\"4\":14321.8896484375,\"5\":14321.8896484375,\"6\":14321.8896484375,\"7\":16971.015625,\"8\":16971.015625,\"9\":16971.015625,\"10\":14321.8896484375,\"11\":14321.8896484375,\"12\":14321.8896484375,\"13\":16971.015625,\"14\":16971.015625,\"15\":14321.8896484375,\"16\":14321.8896484375,\"17\":14321.8896484375,\"18\":16971.015625,\"19\":14321.8896484375,\"20\":16971.015625,\"21\":16971.015625,\"22\":14321.8896484375,\"23\":16971.015625,\"24\":16971.015625,\"25\":14321.8896484375,\"26\":16971.015625,\"27\":14321.8896484375,\"28\":14321.8896484375,\"29\":16971.015625,\"30\":14321.8896484375,\"31\":14321.8896484375,\"32\":16971.015625,\"33\":16971.015625,\"34\":16971.015625,\"35\":16971.015625,\"36\":14321.8896484375,\"37\":16971.015625,\"38\":14321.8896484375,\"39\":14321.8896484375,\"40\":14321.8896484375,\"41\":16971.015625,\"42\":16971.015625,\"43\":16971.015625,\"44\":16971.015625,\"45\":16971.015625,\"46\":16971.015625,\"47\":16971.015625,\"48\":16971.015625,\"49\":16971.015625,\"50\":14321.8896484375,\"51\":14321.8896484375,\"52\":16971.015625,\"53\":16971.015625,\"54\":14321.8896484375,\"55\":16971.015625,\"56\":14321.8896484375,\"57\":16971.015625,\"58\":16971.015625,\"59\":16971.015625,\"60\":16971.015625,\"61\":14321.8896484375,\"62\":16971.015625,\"63\":14321.8896484375,\"64\":14321.8896484375,\"65\":14321.8896484375,\"66\":16971.015625,\"67\":14321.8896484375,\"68\":16971.015625,\"69\":16971.015625,\"70\":14321.8896484375,\"71\":14321.8896484375,\"72\":14321.8896484375,\"73\":14321.8896484375,\"74\":16971.015625,\"75\":16971.015625,\"76\":16971.015625,\"77\":16971.015625,\"78\":14321.8896484375,\"79\":14321.8896484375,\"80\":14321.8896484375,\"81\":14321.8896484375,\"82\":14321.8896484375,\"83\":14321.8896484375,\"84\":16971.015625,\"85\":16971.015625,\"86\":16971.015625,\"87\":16971.015625,\"88\":14321.8896484375,\"89\":16971.015625,\"90\":16971.015625,\"91\":14321.8896484375,\"92\":14321.8896484375,\"93\":16971.015625,\"94\":16971.015625,\"95\":14321.8896484375,\"96\":16971.015625,\"97\":16971.015625,\"98\":14321.8896484375,\"99\":14321.8896484375,\"100\":16971.015625,\"101\":14321.8896484375,\"102\":14321.8896484375,\"103\":14321.8896484375,\"104\":16971.015625,\"105\":14321.8896484375,\"106\":14321.8896484375,\"107\":16971.015625,\"108\":14321.8896484375,\"109\":14321.8896484375,\"110\":14321.8896484375,\"111\":14321.8896484375,\"112\":14321.8896484375,\"113\":14321.8896484375,\"114\":16971.015625,\"115\":16971.015625,\"116\":16971.015625,\"117\":16971.015625,\"118\":16971.015625,\"119\":16971.015625,\"120\":16971.015625,\"121\":16971.015625,\"122\":16971.015625,\"123\":14321.8896484375,\"124\":16971.015625,\"125\":14321.8896484375,\"126\":16971.015625,\"127\":14321.8896484375,\"128\":14321.8896484375,\"129\":14321.8896484375,\"130\":16971.015625,\"131\":14321.8896484375,\"132\":16971.015625,\"133\":16971.015625,\"134\":14321.8896484375,\"135\":16971.015625,\"136\":16971.015625,\"137\":16971.015625,\"138\":16971.015625,\"139\":16971.015625,\"140\":14321.8896484375,\"141\":14321.8896484375,\"142\":14321.8896484375,\"143\":16971.015625,\"144\":14321.8896484375,\"145\":16971.015625,\"146\":16971.015625,\"147\":16971.015625,\"148\":16971.015625,\"149\":14321.8896484375,\"150\":14321.8896484375,\"151\":16971.015625,\"152\":14321.8896484375,\"153\":16971.015625,\"154\":16971.015625,\"155\":16971.015625,\"156\":16971.015625,\"157\":16971.015625,\"158\":14321.8896484375,\"159\":14321.8896484375,\"160\":14321.8896484375,\"161\":16971.015625,\"162\":14321.8896484375,\"163\":14321.8896484375,\"164\":14321.8896484375,\"165\":16971.015625,\"166\":14321.8896484375,\"167\":14321.8896484375,\"168\":14321.8896484375,\"169\":16971.015625,\"170\":14321.8896484375,\"171\":14321.8896484375,\"172\":16971.015625,\"173\":14321.8896484375,\"174\":16971.015625,\"175\":14321.8896484375,\"176\":16971.015625,\"177\":16971.015625,\"178\":16971.015625,\"179\":16971.015625,\"180\":16971.015625,\"181\":16971.015625,\"182\":16971.015625,\"183\":14321.8896484375,\"184\":14321.8896484375,\"185\":16971.015625,\"186\":16971.015625,\"187\":14321.8896484375,\"188\":16971.015625,\"189\":14321.8896484375,\"190\":16971.015625,\"191\":14321.8896484375,\"192\":14321.8896484375,\"193\":16971.015625,\"194\":16971.015625,\"195\":16971.015625,\"196\":16971.015625,\"197\":14321.8896484375,\"198\":16971.015625,\"199\":16971.015625,\"200\":16971.015625,\"201\":14321.8896484375,\"202\":16971.015625,\"203\":14321.8896484375,\"204\":16971.015625,\"205\":16971.015625,\"206\":16971.015625,\"207\":14321.8896484375,\"208\":14321.8896484375,\"209\":16971.015625,\"210\":16971.015625,\"211\":16971.015625,\"212\":16971.015625,\"213\":16971.015625,\"214\":14321.8896484375,\"215\":16971.015625,\"216\":14321.8896484375,\"217\":16971.015625,\"218\":16971.015625,\"219\":16971.015625,\"220\":16971.015625,\"221\":16971.015625,\"222\":16971.015625,\"223\":16971.015625,\"224\":16971.015625,\"225\":16971.015625,\"226\":14321.8896484375,\"227\":16971.015625,\"228\":16971.015625,\"229\":14321.8896484375,\"230\":16971.015625,\"231\":16971.015625,\"232\":16971.015625,\"233\":16971.015625,\"234\":16971.015625,\"235\":14321.8896484375,\"236\":16971.015625,\"237\":14321.8896484375,\"238\":14321.8896484375,\"239\":16971.015625,\"240\":14321.8896484375,\"241\":16971.015625,\"242\":14321.8896484375,\"243\":14321.8896484375,\"244\":16971.015625,\"245\":16971.015625,\"246\":16971.015625,\"247\":16971.015625,\"248\":16971.015625,\"249\":16971.015625,\"250\":16971.015625,\"251\":16971.015625,\"252\":16971.015625,\"253\":14321.8896484375,\"254\":16971.015625,\"255\":14321.8896484375,\"256\":14321.8896484375,\"257\":14321.8896484375,\"258\":14321.8896484375,\"259\":14321.8896484375,\"260\":14321.8896484375,\"261\":14321.8896484375,\"262\":16971.015625,\"263\":16971.015625,\"264\":14321.8896484375,\"265\":14321.8896484375,\"266\":16971.015625,\"267\":16971.015625,\"268\":14321.8896484375,\"269\":14321.8896484375,\"270\":14321.8896484375,\"271\":16971.015625,\"272\":14321.8896484375,\"273\":14321.8896484375,\"274\":16971.015625,\"275\":16971.015625,\"276\":14321.8896484375,\"277\":14321.8896484375,\"278\":14321.8896484375,\"279\":14321.8896484375,\"280\":16971.015625,\"281\":16971.015625,\"282\":16971.015625,\"283\":16971.015625,\"284\":16971.015625,\"285\":14321.8896484375,\"286\":14321.8896484375,\"287\":14321.8896484375,\"288\":14321.8896484375,\"289\":14321.8896484375,\"290\":14321.8896484375,\"291\":16971.015625,\"292\":16971.015625,\"293\":14321.8896484375,\"294\":14321.8896484375,\"295\":16971.015625,\"296\":16971.015625,\"297\":14321.8896484375,\"298\":16971.015625,\"299\":16971.015625,\"300\":16971.015625,\"301\":16971.015625,\"302\":16971.015625,\"303\":16971.015625,\"304\":16971.015625,\"305\":14321.8896484375,\"306\":14321.8896484375,\"307\":16971.015625,\"308\":14321.8896484375,\"309\":14321.8896484375,\"310\":16971.015625,\"311\":16971.015625,\"312\":16971.015625,\"313\":14321.8896484375,\"314\":16971.015625,\"315\":16971.015625,\"316\":14321.8896484375,\"317\":14321.8896484375,\"318\":14321.8896484375,\"319\":16971.015625,\"320\":16971.015625,\"321\":14321.8896484375,\"322\":16971.015625,\"323\":16971.015625,\"324\":16971.015625,\"325\":14321.8896484375,\"326\":16971.015625,\"327\":14321.8896484375,\"328\":16971.015625,\"329\":14321.8896484375,\"330\":14321.8896484375,\"331\":14321.8896484375,\"332\":16971.015625,\"333\":14321.8896484375,\"334\":16971.015625,\"335\":14321.8896484375,\"336\":16971.015625,\"337\":16971.015625,\"338\":16971.015625,\"339\":14321.8896484375,\"340\":16971.015625,\"341\":16971.015625,\"342\":16971.015625,\"343\":14321.8896484375,\"344\":16971.015625,\"345\":14321.8896484375,\"346\":16971.015625,\"347\":16971.015625,\"348\":14321.8896484375,\"349\":16971.015625,\"350\":16971.015625,\"351\":16971.015625,\"352\":14321.8896484375,\"353\":16971.015625,\"354\":16971.015625,\"355\":14321.8896484375,\"356\":16971.015625,\"357\":14321.8896484375,\"358\":14321.8896484375,\"359\":16971.015625,\"360\":16971.015625,\"361\":14321.8896484375,\"362\":16971.015625,\"363\":14321.8896484375,\"364\":16971.015625,\"365\":14321.8896484375,\"366\":14321.8896484375,\"367\":14321.8896484375,\"368\":14321.8896484375,\"369\":14321.8896484375,\"370\":16971.015625,\"371\":16971.015625,\"372\":16971.015625,\"373\":14321.8896484375,\"374\":14321.8896484375,\"375\":14321.8896484375,\"376\":16971.015625,\"377\":14321.8896484375,\"378\":14321.8896484375,\"379\":16971.015625,\"380\":16971.015625,\"381\":14321.8896484375,\"382\":16971.015625,\"383\":14321.8896484375,\"384\":14321.8896484375,\"385\":16971.015625,\"386\":16971.015625,\"387\":16971.015625,\"388\":16971.015625,\"389\":16971.015625,\"390\":14321.8896484375,\"391\":14321.8896484375,\"392\":14321.8896484375,\"393\":14321.8896484375,\"394\":14321.8896484375,\"395\":14321.8896484375,\"396\":14321.8896484375,\"397\":16971.015625,\"398\":16971.015625,\"399\":14321.8896484375,\"400\":16971.015625,\"401\":16971.015625,\"402\":14321.8896484375,\"403\":16971.015625,\"404\":14321.8896484375,\"405\":16971.015625,\"406\":14321.8896484375,\"407\":14321.8896484375,\"408\":14321.8896484375,\"409\":16971.015625,\"410\":14321.8896484375,\"411\":16971.015625,\"412\":16971.015625,\"413\":16971.015625,\"414\":14321.8896484375,\"415\":14321.8896484375,\"416\":16971.015625,\"417\":16971.015625,\"418\":16971.015625,\"419\":16971.015625,\"420\":16971.015625,\"421\":14321.8896484375,\"422\":14321.8896484375,\"423\":16971.015625,\"424\":14321.8896484375,\"425\":14321.8896484375,\"426\":14321.8896484375,\"427\":16971.015625,\"428\":14321.8896484375,\"429\":16971.015625,\"430\":16971.015625,\"431\":14321.8896484375,\"432\":14321.8896484375,\"433\":14321.8896484375,\"434\":14321.8896484375,\"435\":14321.8896484375,\"436\":16971.015625,\"437\":16971.015625,\"438\":16971.015625,\"439\":14321.8896484375,\"440\":14321.8896484375,\"441\":16971.015625,\"442\":14321.8896484375,\"443\":14321.8896484375,\"444\":16971.015625,\"445\":16971.015625,\"446\":16971.015625,\"447\":16971.015625,\"448\":16971.015625,\"449\":16971.015625,\"450\":14321.8896484375,\"451\":14321.8896484375,\"452\":16971.015625,\"453\":14321.8896484375,\"454\":14321.8896484375,\"455\":14321.8896484375,\"456\":14321.8896484375,\"457\":14321.8896484375,\"458\":16971.015625,\"459\":14321.8896484375,\"460\":16971.015625,\"461\":16971.015625,\"462\":14321.8896484375,\"463\":14321.8896484375,\"464\":16971.015625,\"465\":14321.8896484375,\"466\":14321.8896484375,\"467\":16971.015625,\"468\":16971.015625,\"469\":14321.8896484375,\"470\":16971.015625,\"471\":16971.015625,\"472\":16971.015625,\"473\":14321.8896484375,\"474\":16971.015625,\"475\":16971.015625,\"476\":16971.015625,\"477\":14321.8896484375,\"478\":14321.8896484375,\"479\":16971.015625,\"480\":14321.8896484375,\"481\":16971.015625,\"482\":14321.8896484375,\"483\":16971.015625,\"484\":16971.015625,\"485\":14321.8896484375,\"486\":16971.015625,\"487\":16971.015625,\"488\":14321.8896484375,\"489\":16971.015625,\"490\":14321.8896484375,\"491\":16971.015625,\"492\":16971.015625,\"493\":16971.015625,\"494\":14321.8896484375,\"495\":16971.015625,\"496\":16971.015625,\"497\":16971.015625,\"498\":16971.015625,\"499\":14321.8896484375,\"500\":14321.8896484375,\"501\":14321.8896484375,\"502\":16971.015625,\"503\":14321.8896484375,\"504\":14321.8896484375,\"505\":14321.8896484375,\"506\":16971.015625,\"507\":16971.015625,\"508\":16971.015625,\"509\":14321.8896484375,\"510\":16971.015625,\"511\":14321.8896484375,\"512\":14321.8896484375,\"513\":16971.015625,\"514\":14321.8896484375,\"515\":16971.015625,\"516\":16971.015625,\"517\":14321.8896484375,\"518\":14321.8896484375,\"519\":16971.015625,\"520\":16971.015625,\"521\":14321.8896484375,\"522\":16971.015625,\"523\":14321.8896484375,\"524\":14321.8896484375,\"525\":16971.015625,\"526\":16971.015625,\"527\":14321.8896484375,\"528\":16971.015625,\"529\":14321.8896484375,\"530\":16971.015625,\"531\":16971.015625,\"532\":14321.8896484375,\"533\":16971.015625,\"534\":16971.015625,\"535\":16971.015625,\"536\":16971.015625,\"537\":16971.015625,\"538\":14321.8896484375,\"539\":14321.8896484375,\"540\":16971.015625,\"541\":16971.015625,\"542\":14321.8896484375,\"543\":16971.015625,\"544\":14321.8896484375,\"545\":16971.015625,\"546\":16971.015625,\"547\":16971.015625,\"548\":16971.015625,\"549\":16971.015625,\"550\":14321.8896484375,\"551\":16971.015625,\"552\":16971.015625,\"553\":16971.015625,\"554\":14321.8896484375,\"555\":14321.8896484375,\"556\":14321.8896484375,\"557\":14321.8896484375,\"558\":16971.015625,\"559\":16971.015625,\"560\":16971.015625,\"561\":16971.015625,\"562\":14321.8896484375,\"563\":14321.8896484375,\"564\":16971.015625,\"565\":16971.015625,\"566\":16971.015625,\"567\":14321.8896484375,\"568\":14321.8896484375,\"569\":16971.015625,\"570\":16971.015625,\"571\":14321.8896484375,\"572\":16971.015625,\"573\":14321.8896484375,\"574\":14321.8896484375,\"575\":16971.015625,\"576\":14321.8896484375,\"577\":14321.8896484375,\"578\":14321.8896484375,\"579\":16971.015625,\"580\":16971.015625,\"581\":16971.015625,\"582\":16971.015625,\"583\":16971.015625,\"584\":14321.8896484375,\"585\":16971.015625,\"586\":16971.015625,\"587\":16971.015625,\"588\":14321.8896484375,\"589\":14321.8896484375,\"590\":14321.8896484375,\"591\":16971.015625,\"592\":16971.015625,\"593\":16971.015625,\"594\":14321.8896484375,\"595\":14321.8896484375,\"596\":16971.015625,\"597\":14321.8896484375,\"598\":14321.8896484375,\"599\":16971.015625,\"600\":16971.015625,\"601\":16971.015625,\"602\":14321.8896484375,\"603\":16971.015625,\"604\":16971.015625,\"605\":16971.015625,\"606\":16971.015625,\"607\":16971.015625,\"608\":16971.015625,\"609\":16971.015625,\"610\":16971.015625,\"611\":16971.015625,\"612\":14321.8896484375,\"613\":14321.8896484375,\"614\":14321.8896484375,\"615\":16971.015625,\"616\":16971.015625,\"617\":14321.8896484375,\"618\":16971.015625,\"619\":14321.8896484375,\"620\":14321.8896484375,\"621\":14321.8896484375,\"622\":14321.8896484375,\"623\":14321.8896484375,\"624\":16971.015625,\"625\":16971.015625,\"626\":16971.015625,\"627\":14321.8896484375,\"628\":16971.015625,\"629\":16971.015625,\"630\":16971.015625,\"631\":16971.015625,\"632\":14321.8896484375,\"633\":16971.015625,\"634\":14321.8896484375,\"635\":16971.015625,\"636\":16971.015625,\"637\":14321.8896484375,\"638\":16971.015625,\"639\":16971.015625,\"640\":14321.8896484375,\"641\":14321.8896484375,\"642\":16971.015625,\"643\":16971.015625,\"644\":16971.015625,\"645\":14321.8896484375,\"646\":16971.015625,\"647\":14321.8896484375,\"648\":16971.015625,\"649\":14321.8896484375,\"650\":16971.015625,\"651\":16971.015625,\"652\":16971.015625,\"653\":14321.8896484375,\"654\":14321.8896484375,\"655\":16971.015625,\"656\":14321.8896484375,\"657\":14321.8896484375,\"658\":14321.8896484375,\"659\":14321.8896484375,\"660\":14321.8896484375,\"661\":16971.015625,\"662\":14321.8896484375,\"663\":16971.015625,\"664\":16971.015625,\"665\":14321.8896484375,\"666\":14321.8896484375,\"667\":16971.015625,\"668\":14321.8896484375,\"669\":16971.015625,\"670\":14321.8896484375,\"671\":14321.8896484375,\"672\":14321.8896484375,\"673\":14321.8896484375,\"674\":16971.015625,\"675\":16971.015625,\"676\":16971.015625,\"677\":16971.015625,\"678\":16971.015625,\"679\":16971.015625,\"680\":14321.8896484375,\"681\":14321.8896484375,\"682\":16971.015625,\"683\":16971.015625,\"684\":16971.015625,\"685\":14321.8896484375,\"686\":14321.8896484375,\"687\":16971.015625,\"688\":14321.8896484375,\"689\":16971.015625,\"690\":16971.015625,\"691\":16971.015625,\"692\":14321.8896484375,\"693\":14321.8896484375,\"694\":16971.015625,\"695\":14321.8896484375,\"696\":16971.015625,\"697\":16971.015625,\"698\":14321.8896484375,\"699\":14321.8896484375,\"700\":16971.015625,\"701\":14321.8896484375,\"702\":14321.8896484375,\"703\":14321.8896484375,\"704\":14321.8896484375,\"705\":16971.015625,\"706\":14321.8896484375,\"707\":14321.8896484375,\"708\":16971.015625,\"709\":16971.015625,\"710\":16971.015625,\"711\":14321.8896484375,\"712\":14321.8896484375,\"713\":16971.015625,\"714\":14321.8896484375,\"715\":16971.015625,\"716\":16971.015625,\"717\":14321.8896484375,\"718\":14321.8896484375,\"719\":16971.015625,\"720\":14321.8896484375,\"721\":16971.015625,\"722\":14321.8896484375,\"723\":16971.015625,\"724\":14321.8896484375,\"725\":16971.015625,\"726\":14321.8896484375,\"727\":16971.015625,\"728\":16971.015625,\"729\":14321.8896484375,\"730\":16971.015625,\"731\":16971.015625,\"732\":16971.015625,\"733\":16971.015625,\"734\":16971.015625,\"735\":14321.8896484375,\"736\":16971.015625,\"737\":16971.015625,\"738\":14321.8896484375,\"739\":14321.8896484375,\"740\":16971.015625,\"741\":16971.015625,\"742\":14321.8896484375,\"743\":14321.8896484375,\"744\":14321.8896484375,\"745\":16971.015625,\"746\":16971.015625,\"747\":14321.8896484375,\"748\":14321.8896484375,\"749\":14321.8896484375,\"750\":16971.015625,\"751\":14321.8896484375,\"752\":14321.8896484375,\"753\":16971.015625,\"754\":16971.015625,\"755\":16971.015625,\"756\":16971.015625,\"757\":14321.8896484375,\"758\":16971.015625,\"759\":16971.015625,\"760\":16971.015625,\"761\":16971.015625,\"762\":14321.8896484375,\"763\":14321.8896484375,\"764\":14321.8896484375,\"765\":16971.015625,\"766\":14321.8896484375,\"767\":14321.8896484375,\"768\":16971.015625,\"769\":14321.8896484375,\"770\":14321.8896484375,\"771\":16971.015625,\"772\":16971.015625,\"773\":14321.8896484375,\"774\":16971.015625,\"775\":16971.015625,\"776\":16971.015625,\"777\":14321.8896484375,\"778\":16971.015625,\"779\":14321.8896484375,\"780\":16971.015625,\"781\":16971.015625,\"782\":14321.8896484375,\"783\":14321.8896484375,\"784\":14321.8896484375,\"785\":14321.8896484375,\"786\":14321.8896484375,\"787\":16971.015625,\"788\":16971.015625,\"789\":14321.8896484375,\"790\":16971.015625,\"791\":14321.8896484375,\"792\":16971.015625,\"793\":16971.015625,\"794\":16971.015625,\"795\":16971.015625,\"796\":16971.015625,\"797\":14321.8896484375,\"798\":16971.015625,\"799\":16971.015625,\"800\":16971.015625,\"801\":16971.015625,\"802\":16971.015625,\"803\":16971.015625,\"804\":16971.015625,\"805\":16971.015625,\"806\":14321.8896484375,\"807\":14321.8896484375,\"808\":14321.8896484375,\"809\":14321.8896484375,\"810\":14321.8896484375,\"811\":14321.8896484375,\"812\":16971.015625,\"813\":16971.015625,\"814\":14321.8896484375,\"815\":14321.8896484375,\"816\":16971.015625,\"817\":16971.015625,\"818\":16971.015625,\"819\":16971.015625,\"820\":16971.015625,\"821\":14321.8896484375,\"822\":14321.8896484375,\"823\":16971.015625,\"824\":16971.015625,\"825\":16971.015625,\"826\":16971.015625,\"827\":16971.015625,\"828\":16971.015625,\"829\":16971.015625,\"830\":14321.8896484375,\"831\":16971.015625,\"832\":14321.8896484375,\"833\":16971.015625,\"834\":14321.8896484375,\"835\":14321.8896484375,\"836\":16971.015625,\"837\":16971.015625,\"838\":16971.015625,\"839\":16971.015625,\"840\":16971.015625,\"841\":14321.8896484375,\"842\":16971.015625,\"843\":16971.015625,\"844\":14321.8896484375,\"845\":14321.8896484375,\"846\":16971.015625,\"847\":16971.015625,\"848\":16971.015625,\"849\":14321.8896484375,\"850\":16971.015625,\"851\":16971.015625,\"852\":14321.8896484375,\"853\":16971.015625,\"854\":14321.8896484375,\"855\":14321.8896484375,\"856\":16971.015625,\"857\":16971.015625,\"858\":14321.8896484375,\"859\":14321.8896484375,\"860\":14321.8896484375,\"861\":16971.015625,\"862\":14321.8896484375,\"863\":16971.015625,\"864\":14321.8896484375,\"865\":14321.8896484375,\"866\":14321.8896484375,\"867\":16971.015625,\"868\":16971.015625,\"869\":14321.8896484375,\"870\":16971.015625,\"871\":16971.015625,\"872\":14321.8896484375,\"873\":16971.015625,\"874\":14321.8896484375,\"875\":14321.8896484375,\"876\":14321.8896484375,\"877\":14321.8896484375,\"878\":14321.8896484375,\"879\":16971.015625,\"880\":14321.8896484375,\"881\":16971.015625,\"882\":16971.015625,\"883\":16971.015625,\"884\":16971.015625,\"885\":16971.015625,\"886\":14321.8896484375,\"887\":16971.015625,\"888\":16971.015625,\"889\":14321.8896484375,\"890\":14321.8896484375,\"891\":16971.015625,\"892\":14321.8896484375,\"893\":16971.015625,\"894\":16971.015625,\"895\":14321.8896484375,\"896\":16971.015625,\"897\":16971.015625,\"898\":16971.015625,\"899\":16971.015625,\"900\":16971.015625,\"901\":16971.015625,\"902\":16971.015625,\"903\":14321.8896484375,\"904\":14321.8896484375,\"905\":14321.8896484375,\"906\":14321.8896484375,\"907\":16971.015625,\"908\":16971.015625,\"909\":16971.015625,\"910\":14321.8896484375,\"911\":14321.8896484375,\"912\":16971.015625,\"913\":16971.015625,\"914\":16971.015625,\"915\":16971.015625,\"916\":14321.8896484375,\"917\":14321.8896484375,\"918\":14321.8896484375,\"919\":14321.8896484375,\"920\":16971.015625,\"921\":16971.015625,\"922\":16971.015625,\"923\":16971.015625,\"924\":16971.015625,\"925\":16971.015625,\"926\":14321.8896484375,\"927\":16971.015625,\"928\":16971.015625,\"929\":16971.015625,\"930\":16971.015625,\"931\":16971.015625,\"932\":14321.8896484375,\"933\":14321.8896484375,\"934\":16971.015625,\"935\":16971.015625,\"936\":14321.8896484375,\"937\":16971.015625,\"938\":16971.015625,\"939\":14321.8896484375,\"940\":14321.8896484375,\"941\":14321.8896484375,\"942\":16971.015625,\"943\":16971.015625,\"944\":14321.8896484375,\"945\":14321.8896484375,\"946\":16971.015625,\"947\":16971.015625,\"948\":16971.015625,\"949\":16971.015625,\"950\":16971.015625,\"951\":14321.8896484375,\"952\":16971.015625,\"953\":14321.8896484375,\"954\":14321.8896484375,\"955\":14321.8896484375,\"956\":16971.015625,\"957\":16971.015625,\"958\":16971.015625,\"959\":16971.015625,\"960\":14321.8896484375,\"961\":14321.8896484375,\"962\":16971.015625,\"963\":16971.015625,\"964\":16971.015625,\"965\":14321.8896484375,\"966\":16971.015625,\"967\":14321.8896484375,\"968\":14321.8896484375,\"969\":16971.015625,\"970\":16971.015625,\"971\":16971.015625,\"972\":16971.015625,\"973\":16971.015625,\"974\":14321.8896484375,\"975\":16971.015625,\"976\":16971.015625,\"977\":14321.8896484375,\"978\":16971.015625,\"979\":16971.015625,\"980\":14321.8896484375,\"981\":14321.8896484375,\"982\":16971.015625,\"983\":14321.8896484375,\"984\":14321.8896484375,\"985\":16971.015625,\"986\":16971.015625,\"987\":16971.015625,\"988\":14321.8896484375,\"989\":16971.015625,\"990\":14321.8896484375,\"991\":16971.015625,\"992\":16971.015625,\"993\":16971.015625,\"994\":14321.8896484375,\"995\":14321.8896484375,\"996\":16971.015625,\"997\":14321.8896484375,\"998\":16971.015625,\"999\":16971.015625,\"1000\":16971.015625,\"1001\":16971.015625,\"1002\":14321.8896484375,\"1003\":16971.015625,\"1004\":16971.015625,\"1005\":16971.015625,\"1006\":16971.015625,\"1007\":16971.015625,\"1008\":16971.015625,\"1009\":16971.015625,\"1010\":14321.8896484375,\"1011\":16971.015625,\"1012\":16971.015625,\"1013\":14321.8896484375,\"1014\":14321.8896484375,\"1015\":16971.015625,\"1016\":16971.015625,\"1017\":16971.015625,\"1018\":16971.015625,\"1019\":14321.8896484375,\"1020\":16971.015625,\"1021\":16971.015625,\"1022\":16971.015625,\"1023\":16971.015625,\"1024\":14321.8896484375,\"1025\":14321.8896484375,\"1026\":14321.8896484375,\"1027\":16971.015625,\"1028\":14321.8896484375,\"1029\":16971.015625,\"1030\":14321.8896484375,\"1031\":14321.8896484375,\"1032\":14321.8896484375,\"1033\":16971.015625,\"1034\":16971.015625,\"1035\":16971.015625,\"1036\":14321.8896484375,\"1037\":14321.8896484375,\"1038\":16971.015625,\"1039\":16971.015625,\"1040\":16971.015625,\"1041\":14321.8896484375,\"1042\":14321.8896484375,\"1043\":14321.8896484375,\"1044\":14321.8896484375,\"1045\":16971.015625,\"1046\":14321.8896484375,\"1047\":14321.8896484375,\"1048\":16971.015625,\"1049\":16971.015625,\"1050\":14321.8896484375,\"1051\":14321.8896484375,\"1052\":16971.015625,\"1053\":16971.015625,\"1054\":16971.015625,\"1055\":16971.015625,\"1056\":14321.8896484375,\"1057\":14321.8896484375,\"1058\":16971.015625,\"1059\":16971.015625,\"1060\":14321.8896484375,\"1061\":16971.015625,\"1062\":16971.015625,\"1063\":14321.8896484375,\"1064\":14321.8896484375,\"1065\":14321.8896484375,\"1066\":14321.8896484375,\"1067\":16971.015625,\"1068\":16971.015625,\"1069\":14321.8896484375,\"1070\":14321.8896484375,\"1071\":16971.015625,\"1072\":14321.8896484375,\"1073\":14321.8896484375,\"1074\":16971.015625,\"1075\":16971.015625,\"1076\":16971.015625,\"1077\":14321.8896484375,\"1078\":16971.015625,\"1079\":14321.8896484375,\"1080\":16971.015625,\"1081\":14321.8896484375,\"1082\":16971.015625,\"1083\":16971.015625,\"1084\":16971.015625,\"1085\":16971.015625,\"1086\":16971.015625,\"1087\":14321.8896484375,\"1088\":14321.8896484375,\"1089\":16971.015625,\"1090\":16971.015625,\"1091\":16971.015625,\"1092\":14321.8896484375,\"1093\":16971.015625,\"1094\":16971.015625,\"1095\":16971.015625,\"1096\":14321.8896484375,\"1097\":16971.015625,\"1098\":16971.015625,\"1099\":16971.015625,\"1100\":14321.8896484375,\"1101\":16971.015625,\"1102\":14321.8896484375,\"1103\":14321.8896484375,\"1104\":16971.015625,\"1105\":16971.015625,\"1106\":16971.015625,\"1107\":16971.015625,\"1108\":14321.8896484375,\"1109\":16971.015625,\"1110\":16971.015625,\"1111\":14321.8896484375,\"1112\":14321.8896484375,\"1113\":14321.8896484375,\"1114\":14321.8896484375,\"1115\":14321.8896484375,\"1116\":16971.015625,\"1117\":16971.015625,\"1118\":14321.8896484375,\"1119\":14321.8896484375,\"1120\":14321.8896484375,\"1121\":16971.015625,\"1122\":16971.015625,\"1123\":16971.015625,\"1124\":14321.8896484375,\"1125\":14321.8896484375,\"1126\":16971.015625,\"1127\":16971.015625,\"1128\":16971.015625,\"1129\":14321.8896484375,\"1130\":16971.015625,\"1131\":16971.015625,\"1132\":14321.8896484375,\"1133\":16971.015625,\"1134\":16971.015625,\"1135\":16971.015625,\"1136\":14321.8896484375,\"1137\":16971.015625,\"1138\":14321.8896484375,\"1139\":16971.015625,\"1140\":16971.015625,\"1141\":14321.8896484375,\"1142\":14321.8896484375,\"1143\":16971.015625,\"1144\":16971.015625,\"1145\":14321.8896484375,\"1146\":16971.015625,\"1147\":16971.015625,\"1148\":14321.8896484375,\"1149\":16971.015625,\"1150\":14321.8896484375,\"1151\":14321.8896484375,\"1152\":14321.8896484375,\"1153\":16971.015625,\"1154\":16971.015625,\"1155\":14321.8896484375,\"1156\":14321.8896484375,\"1157\":14321.8896484375,\"1158\":16971.015625,\"1159\":16971.015625,\"1160\":16971.015625,\"1161\":14321.8896484375,\"1162\":14321.8896484375,\"1163\":14321.8896484375,\"1164\":16971.015625,\"1165\":16971.015625,\"1166\":14321.8896484375,\"1167\":14321.8896484375,\"1168\":14321.8896484375,\"1169\":16971.015625,\"1170\":14321.8896484375,\"1171\":16971.015625,\"1172\":16971.015625,\"1173\":14321.8896484375,\"1174\":16971.015625,\"1175\":16971.015625,\"1176\":14321.8896484375,\"1177\":16971.015625,\"1178\":14321.8896484375,\"1179\":14321.8896484375,\"1180\":16971.015625,\"1181\":14321.8896484375,\"1182\":14321.8896484375,\"1183\":16971.015625,\"1184\":16971.015625,\"1185\":16971.015625,\"1186\":16971.015625,\"1187\":14321.8896484375,\"1188\":16971.015625,\"1189\":14321.8896484375,\"1190\":14321.8896484375,\"1191\":14321.8896484375,\"1192\":16971.015625,\"1193\":16971.015625,\"1194\":16971.015625,\"1195\":16971.015625,\"1196\":16971.015625,\"1197\":16971.015625,\"1198\":16971.015625,\"1199\":16971.015625,\"1200\":16971.015625,\"1201\":14321.8896484375,\"1202\":14321.8896484375,\"1203\":16971.015625,\"1204\":16971.015625,\"1205\":14321.8896484375,\"1206\":16971.015625,\"1207\":14321.8896484375,\"1208\":16971.015625,\"1209\":16971.015625,\"1210\":16971.015625,\"1211\":16971.015625,\"1212\":14321.8896484375,\"1213\":16971.015625,\"1214\":14321.8896484375,\"1215\":14321.8896484375,\"1216\":14321.8896484375,\"1217\":16971.015625,\"1218\":14321.8896484375,\"1219\":16971.015625,\"1220\":16971.015625,\"1221\":14321.8896484375,\"1222\":14321.8896484375,\"1223\":14321.8896484375,\"1224\":14321.8896484375,\"1225\":16971.015625,\"1226\":16971.015625,\"1227\":16971.015625,\"1228\":16971.015625,\"1229\":14321.8896484375,\"1230\":14321.8896484375,\"1231\":14321.8896484375,\"1232\":14321.8896484375,\"1233\":14321.8896484375,\"1234\":14321.8896484375,\"1235\":16971.015625,\"1236\":16971.015625,\"1237\":16971.015625,\"1238\":16971.015625,\"1239\":14321.8896484375,\"1240\":16971.015625,\"1241\":16971.015625,\"1242\":14321.8896484375,\"1243\":14321.8896484375,\"1244\":16971.015625,\"1245\":16971.015625,\"1246\":14321.8896484375,\"1247\":16971.015625,\"1248\":16971.015625,\"1249\":14321.8896484375,\"1250\":14321.8896484375,\"1251\":16971.015625,\"1252\":14321.8896484375,\"1253\":14321.8896484375,\"1254\":14321.8896484375,\"1255\":16971.015625,\"1256\":14321.8896484375,\"1257\":14321.8896484375,\"1258\":16971.015625,\"1259\":14321.8896484375,\"1260\":14321.8896484375,\"1261\":14321.8896484375,\"1262\":14321.8896484375,\"1263\":14321.8896484375,\"1264\":14321.8896484375,\"1265\":16971.015625,\"1266\":16971.015625,\"1267\":16971.015625,\"1268\":16971.015625,\"1269\":16971.015625,\"1270\":16971.015625,\"1271\":16971.015625,\"1272\":16971.015625,\"1273\":16971.015625,\"1274\":14321.8896484375,\"1275\":16971.015625,\"1276\":14321.8896484375,\"1277\":16971.015625,\"1278\":14321.8896484375,\"1279\":14321.8896484375,\"1280\":14321.8896484375,\"1281\":16971.015625,\"1282\":14321.8896484375,\"1283\":16971.015625,\"1284\":16971.015625,\"1285\":14321.8896484375,\"1286\":16971.015625,\"1287\":16971.015625,\"1288\":16971.015625,\"1289\":16971.015625,\"1290\":16971.015625,\"1291\":14321.8896484375,\"1292\":14321.8896484375,\"1293\":14321.8896484375,\"1294\":16971.015625,\"1295\":14321.8896484375,\"1296\":16971.015625,\"1297\":16971.015625,\"1298\":16971.015625,\"1299\":16971.015625,\"1300\":14321.8896484375,\"1301\":14321.8896484375,\"1302\":16971.015625,\"1303\":14321.8896484375,\"1304\":16971.015625,\"1305\":16971.015625,\"1306\":16971.015625,\"1307\":16971.015625,\"1308\":16971.015625,\"1309\":14321.8896484375,\"1310\":14321.8896484375,\"1311\":14321.8896484375,\"1312\":16971.015625,\"1313\":14321.8896484375,\"1314\":14321.8896484375,\"1315\":14321.8896484375,\"1316\":16971.015625,\"1317\":14321.8896484375,\"1318\":14321.8896484375,\"1319\":14321.8896484375,\"1320\":16971.015625,\"1321\":14321.8896484375,\"1322\":14321.8896484375,\"1323\":16971.015625,\"1324\":14321.8896484375,\"1325\":16971.015625,\"1326\":14321.8896484375,\"1327\":16971.015625,\"1328\":16971.015625,\"1329\":16971.015625,\"1330\":16971.015625,\"1331\":16971.015625,\"1332\":16971.015625,\"1333\":16971.015625,\"1334\":14321.8896484375,\"1335\":14321.8896484375,\"1336\":16971.015625,\"1337\":16971.015625,\"1338\":14321.8896484375,\"1339\":16971.015625,\"1340\":14321.8896484375,\"1341\":16971.015625,\"1342\":14321.8896484375,\"1343\":14321.8896484375,\"1344\":16971.015625,\"1345\":16971.015625,\"1346\":16971.015625,\"1347\":16971.015625,\"1348\":14321.8896484375,\"1349\":16971.015625,\"1350\":16971.015625,\"1351\":16971.015625,\"1352\":14321.8896484375,\"1353\":16971.015625,\"1354\":14321.8896484375,\"1355\":16971.015625,\"1356\":16971.015625,\"1357\":16971.015625,\"1358\":14321.8896484375,\"1359\":14321.8896484375,\"1360\":16971.015625,\"1361\":16971.015625,\"1362\":16971.015625,\"1363\":16971.015625,\"1364\":16971.015625,\"1365\":14321.8896484375,\"1366\":16971.015625,\"1367\":14321.8896484375,\"1368\":16971.015625,\"1369\":16971.015625,\"1370\":16971.015625,\"1371\":16971.015625,\"1372\":16971.015625,\"1373\":16971.015625,\"1374\":16971.015625,\"1375\":16971.015625,\"1376\":16971.015625,\"1377\":14321.8896484375,\"1378\":16971.015625,\"1379\":16971.015625,\"1380\":14321.8896484375,\"1381\":16971.015625,\"1382\":16971.015625,\"1383\":16971.015625,\"1384\":16971.015625,\"1385\":16971.015625,\"1386\":14321.8896484375,\"1387\":16971.015625,\"1388\":14321.8896484375,\"1389\":14321.8896484375,\"1390\":16971.015625,\"1391\":14321.8896484375,\"1392\":16971.015625,\"1393\":14321.8896484375,\"1394\":14321.8896484375,\"1395\":16971.015625,\"1396\":16971.015625,\"1397\":16971.015625,\"1398\":16971.015625,\"1399\":16971.015625,\"1400\":16971.015625,\"1401\":16971.015625,\"1402\":16971.015625,\"1403\":16971.015625,\"1404\":14321.8896484375,\"1405\":16971.015625,\"1406\":14321.8896484375,\"1407\":14321.8896484375,\"1408\":14321.8896484375,\"1409\":14321.8896484375,\"1410\":14321.8896484375,\"1411\":14321.8896484375,\"1412\":14321.8896484375,\"1413\":16971.015625,\"1414\":16971.015625,\"1415\":14321.8896484375,\"1416\":14321.8896484375,\"1417\":16971.015625,\"1418\":16971.015625,\"1419\":14321.8896484375,\"1420\":14321.8896484375,\"1421\":14321.8896484375,\"1422\":16971.015625,\"1423\":14321.8896484375,\"1424\":14321.8896484375,\"1425\":16971.015625,\"1426\":16971.015625,\"1427\":14321.8896484375,\"1428\":14321.8896484375,\"1429\":14321.8896484375,\"1430\":14321.8896484375,\"1431\":16971.015625,\"1432\":16971.015625,\"1433\":16971.015625,\"1434\":16971.015625,\"1435\":16971.015625,\"1436\":14321.8896484375,\"1437\":14321.8896484375,\"1438\":14321.8896484375,\"1439\":14321.8896484375,\"1440\":14321.8896484375,\"1441\":14321.8896484375,\"1442\":16971.015625,\"1443\":16971.015625,\"1444\":14321.8896484375,\"1445\":14321.8896484375,\"1446\":16971.015625,\"1447\":16971.015625,\"1448\":14321.8896484375,\"1449\":16971.015625,\"1450\":16971.015625,\"1451\":16971.015625,\"1452\":16971.015625,\"1453\":16971.015625,\"1454\":16971.015625,\"1455\":16971.015625,\"1456\":14321.8896484375,\"1457\":14321.8896484375,\"1458\":16971.015625,\"1459\":14321.8896484375,\"1460\":14321.8896484375,\"1461\":16971.015625,\"1462\":16971.015625,\"1463\":16971.015625,\"1464\":14321.8896484375,\"1465\":16971.015625,\"1466\":16971.015625,\"1467\":14321.8896484375,\"1468\":14321.8896484375,\"1469\":14321.8896484375,\"1470\":16971.015625,\"1471\":16971.015625,\"1472\":14321.8896484375,\"1473\":16971.015625,\"1474\":16971.015625,\"1475\":16971.015625,\"1476\":14321.8896484375,\"1477\":16971.015625,\"1478\":14321.8896484375,\"1479\":16971.015625,\"1480\":14321.8896484375,\"1481\":14321.8896484375,\"1482\":14321.8896484375,\"1483\":16971.015625,\"1484\":14321.8896484375,\"1485\":16971.015625,\"1486\":14321.8896484375,\"1487\":16971.015625,\"1488\":16971.015625,\"1489\":16971.015625,\"1490\":14321.8896484375,\"1491\":16971.015625,\"1492\":16971.015625,\"1493\":16971.015625,\"1494\":14321.8896484375,\"1495\":16971.015625,\"1496\":14321.8896484375,\"1497\":16971.015625,\"1498\":16971.015625,\"1499\":14321.8896484375,\"1500\":16971.015625,\"1501\":16971.015625,\"1502\":16971.015625,\"1503\":14321.8896484375,\"1504\":16971.015625,\"1505\":16971.015625,\"1506\":14321.8896484375,\"1507\":16971.015625,\"1508\":14321.8896484375,\"1509\":14321.8896484375,\"1510\":16971.015625,\"1511\":16971.015625,\"1512\":14321.8896484375,\"1513\":16971.015625,\"1514\":14321.8896484375,\"1515\":16971.015625,\"1516\":14321.8896484375,\"1517\":14321.8896484375,\"1518\":14321.8896484375,\"1519\":14321.8896484375,\"1520\":14321.8896484375,\"1521\":16971.015625,\"1522\":16971.015625,\"1523\":16971.015625,\"1524\":14321.8896484375,\"1525\":14321.8896484375,\"1526\":14321.8896484375,\"1527\":16971.015625,\"1528\":14321.8896484375,\"1529\":14321.8896484375,\"1530\":16971.015625,\"1531\":16971.015625,\"1532\":14321.8896484375,\"1533\":16971.015625,\"1534\":14321.8896484375,\"1535\":14321.8896484375,\"1536\":16971.015625,\"1537\":16971.015625,\"1538\":16971.015625,\"1539\":16971.015625,\"1540\":16971.015625,\"1541\":14321.8896484375,\"1542\":14321.8896484375,\"1543\":14321.8896484375,\"1544\":14321.8896484375,\"1545\":14321.8896484375,\"1546\":14321.8896484375,\"1547\":14321.8896484375,\"1548\":16971.015625,\"1549\":16971.015625,\"1550\":14321.8896484375,\"1551\":16971.015625,\"1552\":16971.015625,\"1553\":14321.8896484375,\"1554\":16971.015625,\"1555\":14321.8896484375,\"1556\":16971.015625,\"1557\":14321.8896484375,\"1558\":14321.8896484375,\"1559\":14321.8896484375,\"1560\":16971.015625,\"1561\":14321.8896484375,\"1562\":16971.015625,\"1563\":16971.015625,\"1564\":16971.015625,\"1565\":14321.8896484375,\"1566\":14321.8896484375,\"1567\":16971.015625,\"1568\":16971.015625,\"1569\":16971.015625,\"1570\":16971.015625,\"1571\":16971.015625,\"1572\":14321.8896484375,\"1573\":14321.8896484375,\"1574\":16971.015625,\"1575\":14321.8896484375,\"1576\":14321.8896484375,\"1577\":14321.8896484375,\"1578\":16971.015625,\"1579\":14321.8896484375,\"1580\":16971.015625,\"1581\":16971.015625,\"1582\":14321.8896484375,\"1583\":14321.8896484375,\"1584\":14321.8896484375,\"1585\":14321.8896484375,\"1586\":14321.8896484375,\"1587\":16971.015625,\"1588\":16971.015625,\"1589\":16971.015625,\"1590\":14321.8896484375,\"1591\":14321.8896484375,\"1592\":16971.015625,\"1593\":14321.8896484375,\"1594\":14321.8896484375,\"1595\":16971.015625,\"1596\":16971.015625,\"1597\":16971.015625,\"1598\":16971.015625,\"1599\":16971.015625,\"1600\":16971.015625,\"1601\":14321.8896484375,\"1602\":14321.8896484375,\"1603\":16971.015625,\"1604\":14321.8896484375,\"1605\":14321.8896484375,\"1606\":14321.8896484375,\"1607\":14321.8896484375,\"1608\":14321.8896484375,\"1609\":16971.015625,\"1610\":14321.8896484375,\"1611\":16971.015625,\"1612\":16971.015625,\"1613\":14321.8896484375,\"1614\":14321.8896484375,\"1615\":16971.015625,\"1616\":14321.8896484375,\"1617\":14321.8896484375,\"1618\":16971.015625,\"1619\":16971.015625,\"1620\":14321.8896484375,\"1621\":16971.015625,\"1622\":16971.015625,\"1623\":16971.015625,\"1624\":14321.8896484375,\"1625\":16971.015625,\"1626\":16971.015625,\"1627\":16971.015625,\"1628\":14321.8896484375,\"1629\":14321.8896484375,\"1630\":16971.015625,\"1631\":14321.8896484375,\"1632\":16971.015625,\"1633\":14321.8896484375,\"1634\":16971.015625,\"1635\":16971.015625,\"1636\":14321.8896484375,\"1637\":16971.015625,\"1638\":16971.015625,\"1639\":14321.8896484375,\"1640\":16971.015625,\"1641\":14321.8896484375,\"1642\":16971.015625,\"1643\":16971.015625,\"1644\":16971.015625,\"1645\":14321.8896484375,\"1646\":16971.015625,\"1647\":16971.015625,\"1648\":16971.015625,\"1649\":16971.015625,\"1650\":14321.8896484375,\"1651\":14321.8896484375,\"1652\":14321.8896484375,\"1653\":16971.015625,\"1654\":14321.8896484375,\"1655\":14321.8896484375,\"1656\":14321.8896484375,\"1657\":16971.015625,\"1658\":16971.015625,\"1659\":16971.015625,\"1660\":14321.8896484375,\"1661\":16971.015625,\"1662\":14321.8896484375,\"1663\":14321.8896484375,\"1664\":16971.015625,\"1665\":14321.8896484375,\"1666\":16971.015625,\"1667\":16971.015625,\"1668\":14321.8896484375,\"1669\":14321.8896484375,\"1670\":16971.015625,\"1671\":16971.015625,\"1672\":14321.8896484375,\"1673\":16971.015625,\"1674\":14321.8896484375,\"1675\":14321.8896484375,\"1676\":16971.015625,\"1677\":16971.015625,\"1678\":14321.8896484375,\"1679\":16971.015625,\"1680\":14321.8896484375,\"1681\":16971.015625,\"1682\":16971.015625,\"1683\":14321.8896484375,\"1684\":16971.015625,\"1685\":16971.015625,\"1686\":16971.015625,\"1687\":16971.015625,\"1688\":16971.015625,\"1689\":14321.8896484375,\"1690\":14321.8896484375,\"1691\":16971.015625,\"1692\":16971.015625,\"1693\":14321.8896484375,\"1694\":16971.015625,\"1695\":14321.8896484375,\"1696\":16971.015625,\"1697\":16971.015625,\"1698\":16971.015625,\"1699\":16971.015625,\"1700\":16971.015625,\"1701\":14321.8896484375,\"1702\":16971.015625,\"1703\":16971.015625,\"1704\":16971.015625,\"1705\":14321.8896484375,\"1706\":14321.8896484375,\"1707\":14321.8896484375,\"1708\":14321.8896484375,\"1709\":16971.015625,\"1710\":16971.015625,\"1711\":16971.015625,\"1712\":16971.015625,\"1713\":14321.8896484375,\"1714\":14321.8896484375,\"1715\":16971.015625,\"1716\":16971.015625,\"1717\":16971.015625,\"1718\":14321.8896484375,\"1719\":14321.8896484375,\"1720\":16971.015625,\"1721\":16971.015625,\"1722\":14321.8896484375,\"1723\":16971.015625,\"1724\":14321.8896484375,\"1725\":14321.8896484375,\"1726\":16971.015625,\"1727\":14321.8896484375,\"1728\":14321.8896484375,\"1729\":14321.8896484375,\"1730\":16971.015625,\"1731\":16971.015625,\"1732\":16971.015625,\"1733\":16971.015625,\"1734\":16971.015625,\"1735\":14321.8896484375,\"1736\":16971.015625,\"1737\":16971.015625,\"1738\":16971.015625,\"1739\":14321.8896484375,\"1740\":14321.8896484375,\"1741\":14321.8896484375,\"1742\":16971.015625,\"1743\":16971.015625,\"1744\":16971.015625,\"1745\":14321.8896484375,\"1746\":14321.8896484375,\"1747\":16971.015625,\"1748\":14321.8896484375,\"1749\":14321.8896484375,\"1750\":16971.015625,\"1751\":16971.015625,\"1752\":16971.015625,\"1753\":14321.8896484375,\"1754\":16971.015625,\"1755\":16971.015625,\"1756\":16971.015625,\"1757\":16971.015625,\"1758\":16971.015625,\"1759\":16971.015625,\"1760\":16971.015625,\"1761\":16971.015625,\"1762\":16971.015625,\"1763\":14321.8896484375,\"1764\":14321.8896484375,\"1765\":14321.8896484375,\"1766\":16971.015625,\"1767\":16971.015625,\"1768\":14321.8896484375,\"1769\":16971.015625,\"1770\":14321.8896484375,\"1771\":14321.8896484375,\"1772\":14321.8896484375,\"1773\":14321.8896484375,\"1774\":14321.8896484375,\"1775\":16971.015625,\"1776\":16971.015625,\"1777\":16971.015625,\"1778\":14321.8896484375,\"1779\":16971.015625,\"1780\":16971.015625,\"1781\":16971.015625,\"1782\":16971.015625,\"1783\":14321.8896484375,\"1784\":16971.015625,\"1785\":14321.8896484375,\"1786\":16971.015625,\"1787\":16971.015625,\"1788\":14321.8896484375,\"1789\":16971.015625,\"1790\":16971.015625,\"1791\":14321.8896484375,\"1792\":14321.8896484375,\"1793\":16971.015625,\"1794\":16971.015625,\"1795\":16971.015625,\"1796\":14321.8896484375,\"1797\":16971.015625,\"1798\":14321.8896484375,\"1799\":16971.015625,\"1800\":14321.8896484375,\"1801\":16971.015625,\"1802\":16971.015625,\"1803\":16971.015625,\"1804\":14321.8896484375,\"1805\":14321.8896484375,\"1806\":16971.015625,\"1807\":14321.8896484375,\"1808\":14321.8896484375,\"1809\":14321.8896484375,\"1810\":14321.8896484375,\"1811\":14321.8896484375,\"1812\":16971.015625,\"1813\":14321.8896484375,\"1814\":16971.015625,\"1815\":16971.015625,\"1816\":14321.8896484375,\"1817\":14321.8896484375,\"1818\":16971.015625,\"1819\":14321.8896484375,\"1820\":16971.015625,\"1821\":14321.8896484375,\"1822\":14321.8896484375,\"1823\":14321.8896484375,\"1824\":14321.8896484375,\"1825\":16971.015625,\"1826\":16971.015625,\"1827\":16971.015625,\"1828\":16971.015625,\"1829\":16971.015625,\"1830\":16971.015625,\"1831\":14321.8896484375,\"1832\":14321.8896484375,\"1833\":16971.015625,\"1834\":16971.015625,\"1835\":16971.015625,\"1836\":14321.8896484375,\"1837\":14321.8896484375,\"1838\":16971.015625,\"1839\":14321.8896484375,\"1840\":16971.015625,\"1841\":16971.015625,\"1842\":16971.015625,\"1843\":14321.8896484375,\"1844\":14321.8896484375,\"1845\":16971.015625,\"1846\":14321.8896484375,\"1847\":16971.015625,\"1848\":16971.015625,\"1849\":14321.8896484375,\"1850\":14321.8896484375,\"1851\":16971.015625,\"1852\":14321.8896484375,\"1853\":14321.8896484375,\"1854\":14321.8896484375,\"1855\":14321.8896484375,\"1856\":16971.015625,\"1857\":14321.8896484375,\"1858\":14321.8896484375,\"1859\":16971.015625,\"1860\":16971.015625,\"1861\":16971.015625,\"1862\":14321.8896484375,\"1863\":14321.8896484375,\"1864\":16971.015625,\"1865\":14321.8896484375,\"1866\":16971.015625,\"1867\":16971.015625,\"1868\":14321.8896484375,\"1869\":14321.8896484375,\"1870\":16971.015625,\"1871\":14321.8896484375,\"1872\":16971.015625,\"1873\":14321.8896484375,\"1874\":16971.015625,\"1875\":14321.8896484375,\"1876\":16971.015625,\"1877\":14321.8896484375,\"1878\":16971.015625,\"1879\":16971.015625,\"1880\":14321.8896484375,\"1881\":16971.015625,\"1882\":16971.015625,\"1883\":16971.015625,\"1884\":16971.015625,\"1885\":16971.015625,\"1886\":14321.8896484375,\"1887\":16971.015625,\"1888\":16971.015625,\"1889\":14321.8896484375,\"1890\":14321.8896484375,\"1891\":16971.015625,\"1892\":16971.015625,\"1893\":14321.8896484375,\"1894\":14321.8896484375,\"1895\":14321.8896484375,\"1896\":16971.015625,\"1897\":16971.015625,\"1898\":14321.8896484375,\"1899\":14321.8896484375,\"1900\":14321.8896484375,\"1901\":16971.015625,\"1902\":14321.8896484375,\"1903\":14321.8896484375,\"1904\":16971.015625,\"1905\":16971.015625,\"1906\":16971.015625,\"1907\":16971.015625,\"1908\":14321.8896484375,\"1909\":16971.015625,\"1910\":16971.015625,\"1911\":16971.015625,\"1912\":16971.015625,\"1913\":14321.8896484375,\"1914\":14321.8896484375,\"1915\":14321.8896484375,\"1916\":16971.015625,\"1917\":14321.8896484375,\"1918\":14321.8896484375,\"1919\":16971.015625,\"1920\":14321.8896484375,\"1921\":14321.8896484375,\"1922\":16971.015625,\"1923\":16971.015625,\"1924\":14321.8896484375,\"1925\":16971.015625,\"1926\":16971.015625,\"1927\":16971.015625,\"1928\":14321.8896484375,\"1929\":16971.015625,\"1930\":14321.8896484375,\"1931\":16971.015625,\"1932\":16971.015625,\"1933\":14321.8896484375,\"1934\":14321.8896484375,\"1935\":14321.8896484375,\"1936\":14321.8896484375,\"1937\":14321.8896484375,\"1938\":16971.015625,\"1939\":16971.015625,\"1940\":14321.8896484375,\"1941\":16971.015625,\"1942\":14321.8896484375,\"1943\":16971.015625,\"1944\":16971.015625,\"1945\":16971.015625,\"1946\":16971.015625,\"1947\":16971.015625,\"1948\":14321.8896484375,\"1949\":16971.015625,\"1950\":16971.015625,\"1951\":16971.015625,\"1952\":16971.015625,\"1953\":16971.015625,\"1954\":16971.015625,\"1955\":16971.015625,\"1956\":16971.015625,\"1957\":14321.8896484375,\"1958\":14321.8896484375,\"1959\":14321.8896484375,\"1960\":14321.8896484375,\"1961\":14321.8896484375,\"1962\":14321.8896484375,\"1963\":16971.015625,\"1964\":16971.015625,\"1965\":14321.8896484375,\"1966\":14321.8896484375,\"1967\":16971.015625,\"1968\":16971.015625,\"1969\":16971.015625,\"1970\":16971.015625,\"1971\":16971.015625,\"1972\":14321.8896484375,\"1973\":14321.8896484375,\"1974\":16971.015625,\"1975\":16971.015625,\"1976\":16971.015625,\"1977\":16971.015625,\"1978\":16971.015625,\"1979\":16971.015625,\"1980\":16971.015625,\"1981\":14321.8896484375,\"1982\":16971.015625,\"1983\":14321.8896484375,\"1984\":16971.015625,\"1985\":14321.8896484375,\"1986\":14321.8896484375,\"1987\":16971.015625,\"1988\":16971.015625,\"1989\":16971.015625,\"1990\":16971.015625,\"1991\":16971.015625,\"1992\":14321.8896484375,\"1993\":16971.015625,\"1994\":16971.015625,\"1995\":14321.8896484375,\"1996\":14321.8896484375,\"1997\":16971.015625,\"1998\":16971.015625,\"1999\":16971.015625,\"2000\":14321.8896484375,\"2001\":16971.015625,\"2002\":16971.015625,\"2003\":14321.8896484375,\"2004\":16971.015625,\"2005\":14321.8896484375,\"2006\":14321.8896484375,\"2007\":16971.015625,\"2008\":16971.015625,\"2009\":14321.8896484375,\"2010\":14321.8896484375,\"2011\":14321.8896484375,\"2012\":16971.015625,\"2013\":14321.8896484375,\"2014\":16971.015625,\"2015\":14321.8896484375,\"2016\":14321.8896484375,\"2017\":14321.8896484375,\"2018\":16971.015625,\"2019\":16971.015625,\"2020\":14321.8896484375,\"2021\":16971.015625,\"2022\":16971.015625,\"2023\":14321.8896484375,\"2024\":16971.015625,\"2025\":14321.8896484375,\"2026\":14321.8896484375,\"2027\":14321.8896484375,\"2028\":14321.8896484375,\"2029\":14321.8896484375,\"2030\":16971.015625,\"2031\":14321.8896484375,\"2032\":16971.015625,\"2033\":16971.015625,\"2034\":16971.015625,\"2035\":16971.015625,\"2036\":16971.015625,\"2037\":14321.8896484375,\"2038\":16971.015625,\"2039\":16971.015625,\"2040\":14321.8896484375,\"2041\":14321.8896484375,\"2042\":16971.015625,\"2043\":14321.8896484375,\"2044\":16971.015625,\"2045\":16971.015625,\"2046\":14321.8896484375,\"2047\":16971.015625,\"2048\":16971.015625,\"2049\":16971.015625,\"2050\":16971.015625,\"2051\":16971.015625,\"2052\":16971.015625,\"2053\":16971.015625,\"2054\":14321.8896484375,\"2055\":14321.8896484375,\"2056\":14321.8896484375,\"2057\":14321.8896484375,\"2058\":16971.015625,\"2059\":16971.015625,\"2060\":16971.015625,\"2061\":14321.8896484375,\"2062\":14321.8896484375,\"2063\":16971.015625,\"2064\":16971.015625,\"2065\":16971.015625,\"2066\":16971.015625,\"2067\":14321.8896484375,\"2068\":14321.8896484375,\"2069\":14321.8896484375,\"2070\":14321.8896484375,\"2071\":16971.015625,\"2072\":16971.015625,\"2073\":16971.015625,\"2074\":16971.015625,\"2075\":16971.015625,\"2076\":16971.015625,\"2077\":14321.8896484375,\"2078\":16971.015625,\"2079\":16971.015625,\"2080\":16971.015625,\"2081\":16971.015625,\"2082\":16971.015625,\"2083\":14321.8896484375,\"2084\":14321.8896484375,\"2085\":16971.015625,\"2086\":16971.015625,\"2087\":14321.8896484375,\"2088\":16971.015625,\"2089\":16971.015625,\"2090\":14321.8896484375,\"2091\":14321.8896484375,\"2092\":14321.8896484375,\"2093\":16971.015625,\"2094\":16971.015625,\"2095\":14321.8896484375,\"2096\":14321.8896484375,\"2097\":16971.015625,\"2098\":16971.015625,\"2099\":16971.015625,\"2100\":16971.015625,\"2101\":16971.015625,\"2102\":14321.8896484375,\"2103\":16971.015625,\"2104\":14321.8896484375,\"2105\":14321.8896484375,\"2106\":14321.8896484375,\"2107\":16971.015625,\"2108\":16971.015625,\"2109\":16971.015625,\"2110\":16971.015625,\"2111\":14321.8896484375,\"2112\":14321.8896484375,\"2113\":16971.015625,\"2114\":16971.015625,\"2115\":16971.015625,\"2116\":14321.8896484375,\"2117\":16971.015625,\"2118\":14321.8896484375,\"2119\":14321.8896484375,\"2120\":16971.015625,\"2121\":16971.015625,\"2122\":16971.015625,\"2123\":16971.015625,\"2124\":16971.015625,\"2125\":14321.8896484375,\"2126\":16971.015625,\"2127\":16971.015625,\"2128\":14321.8896484375,\"2129\":16971.015625,\"2130\":16971.015625,\"2131\":14321.8896484375,\"2132\":14321.8896484375,\"2133\":16971.015625,\"2134\":14321.8896484375,\"2135\":14321.8896484375,\"2136\":16971.015625,\"2137\":16971.015625,\"2138\":16971.015625,\"2139\":14321.8896484375,\"2140\":16971.015625,\"2141\":14321.8896484375,\"2142\":16971.015625,\"2143\":16971.015625,\"2144\":16971.015625,\"2145\":14321.8896484375,\"2146\":14321.8896484375,\"2147\":16971.015625,\"2148\":14321.8896484375,\"2149\":16971.015625,\"2150\":16971.015625,\"2151\":16971.015625,\"2152\":16971.015625,\"2153\":14321.8896484375,\"2154\":16971.015625,\"2155\":16971.015625,\"2156\":16971.015625,\"2157\":16971.015625,\"2158\":16971.015625,\"2159\":16971.015625,\"2160\":16971.015625,\"2161\":14321.8896484375,\"2162\":16971.015625,\"2163\":16971.015625,\"2164\":14321.8896484375,\"2165\":14321.8896484375,\"2166\":16971.015625,\"2167\":16971.015625,\"2168\":16971.015625,\"2169\":16971.015625,\"2170\":14321.8896484375,\"2171\":16971.015625,\"2172\":16971.015625,\"2173\":16971.015625,\"2174\":16971.015625,\"2175\":14321.8896484375,\"2176\":14321.8896484375,\"2177\":14321.8896484375,\"2178\":16971.015625,\"2179\":14321.8896484375,\"2180\":16971.015625,\"2181\":14321.8896484375,\"2182\":14321.8896484375,\"2183\":14321.8896484375,\"2184\":16971.015625,\"2185\":16971.015625,\"2186\":16971.015625,\"2187\":14321.8896484375,\"2188\":14321.8896484375,\"2189\":16971.015625,\"2190\":16971.015625,\"2191\":16971.015625,\"2192\":14321.8896484375,\"2193\":14321.8896484375,\"2194\":14321.8896484375,\"2195\":14321.8896484375,\"2196\":16971.015625,\"2197\":14321.8896484375,\"2198\":14321.8896484375,\"2199\":16971.015625,\"2200\":16971.015625,\"2201\":14321.8896484375,\"2202\":14321.8896484375,\"2203\":16971.015625,\"2204\":16971.015625,\"2205\":16971.015625,\"2206\":16971.015625,\"2207\":14321.8896484375,\"2208\":14321.8896484375,\"2209\":16971.015625,\"2210\":16971.015625,\"2211\":14321.8896484375,\"2212\":16971.015625,\"2213\":16971.015625,\"2214\":14321.8896484375,\"2215\":14321.8896484375,\"2216\":14321.8896484375,\"2217\":14321.8896484375,\"2218\":16971.015625,\"2219\":16971.015625,\"2220\":14321.8896484375,\"2221\":14321.8896484375,\"2222\":16971.015625,\"2223\":14321.8896484375,\"2224\":14321.8896484375,\"2225\":16971.015625,\"2226\":16971.015625,\"2227\":16971.015625,\"2228\":14321.8896484375,\"2229\":16971.015625,\"2230\":14321.8896484375,\"2231\":16971.015625,\"2232\":14321.8896484375,\"2233\":16971.015625,\"2234\":16971.015625,\"2235\":16971.015625,\"2236\":16971.015625,\"2237\":16971.015625,\"2238\":14321.8896484375,\"2239\":14321.8896484375,\"2240\":16971.015625,\"2241\":16971.015625,\"2242\":16971.015625,\"2243\":14321.8896484375,\"2244\":16971.015625,\"2245\":16971.015625,\"2246\":16971.015625,\"2247\":14321.8896484375,\"2248\":16971.015625,\"2249\":16971.015625,\"2250\":16971.015625,\"2251\":14321.8896484375,\"2252\":16971.015625,\"2253\":14321.8896484375,\"2254\":14321.8896484375,\"2255\":16971.015625,\"2256\":16971.015625,\"2257\":16971.015625,\"2258\":16971.015625,\"2259\":14321.8896484375,\"2260\":16971.015625,\"2261\":16971.015625,\"2262\":14321.8896484375,\"2263\":14321.8896484375,\"2264\":14321.8896484375,\"2265\":14321.8896484375,\"2266\":14321.8896484375,\"2267\":16971.015625,\"2268\":16971.015625,\"2269\":14321.8896484375,\"2270\":14321.8896484375,\"2271\":14321.8896484375,\"2272\":16971.015625,\"2273\":16971.015625,\"2274\":16971.015625,\"2275\":14321.8896484375,\"2276\":14321.8896484375,\"2277\":16971.015625,\"2278\":16971.015625,\"2279\":16971.015625,\"2280\":14321.8896484375,\"2281\":16971.015625,\"2282\":16971.015625,\"2283\":14321.8896484375,\"2284\":16971.015625,\"2285\":16971.015625,\"2286\":16971.015625,\"2287\":14321.8896484375,\"2288\":16971.015625,\"2289\":14321.8896484375,\"2290\":16971.015625,\"2291\":16971.015625,\"2292\":14321.8896484375,\"2293\":14321.8896484375,\"2294\":16971.015625,\"2295\":16971.015625,\"2296\":14321.8896484375,\"2297\":16971.015625,\"2298\":16971.015625,\"2299\":14321.8896484375,\"2300\":16971.015625,\"2301\":14321.8896484375,\"2302\":14321.8896484375}}\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import pickle\n",
        "\n",
        "# Define the function for prediction\n",
        "def FunctionPredictResult(InputData):\n",
        "    # Load necessary libraries and models\n",
        "    with open('Final_XGB_Model.pkl', 'rb') as fileReadStream:\n",
        "        PredictionModel = pickle.load(fileReadStream)\n",
        "\n",
        "    # Load the data used for model training\n",
        "    DataForML = pd.read_pickle('DataForML.pkl')\n",
        "\n",
        "    # Combine the new input data with the training data\n",
        "    InputData = pd.concat([InputData, DataForML], ignore_index=True)\n",
        "\n",
        "    # Ensure that the input data has the same columns as it was used for training\n",
        "    Predictors = ['age', 'children']\n",
        "\n",
        "    # Extract the relevant features and generate dummy variables if necessary\n",
        "    InputData = InputData[Predictors]\n",
        "\n",
        "    # If there are nominal variables requiring dummy encoding, you can apply pd.get_dummies here\n",
        "\n",
        "    # Assuming PredictorScalerFit is defined elsewhere and used for standardization\n",
        "    # X = PredictorScalerFit.transform(InputData)\n",
        "\n",
        "    # Generate predictions\n",
        "    Predictions = PredictionModel.predict(InputData)\n",
        "\n",
        "    # Create a DataFrame to store the predictions\n",
        "    PredictionResult = pd.DataFrame(Predictions, columns=['Prediction'])\n",
        "\n",
        "    return PredictionResult\n",
        "\n",
        "# Creating the function which can take inputs and return prediction\n",
        "def FunctionGeneratePrediction(inp_age, inp_children):\n",
        "    # Creating a DataFrame for the model input\n",
        "    SampleInputData = pd.DataFrame(data=[[inp_age, inp_children]], columns=['age', 'children'])\n",
        "\n",
        "    # Calling the function defined above using the input parameters\n",
        "    Predictions = FunctionPredictResult(InputData=SampleInputData)\n",
        "\n",
        "    # Returning the predictions\n",
        "    return Predictions.to_json()\n",
        "\n",
        "# Function call\n",
        "result = FunctionGeneratePrediction(inp_age=21, inp_children=0)\n",
        "print(result)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 122
        },
        "id": "p6kBUYOfU53Z",
        "outputId": "6310cc57-3407-4287-9593-5cf624be5d1e"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'{\"Prediction\":{\"0\":14321.8896484375,\"1\":14321.8896484375,\"2\":16971.015625,\"3\":16971.015625,\"4\":14321.8896484375,\"5\":14321.8896484375,\"6\":14321.8896484375,\"7\":16971.015625,\"8\":16971.015625,\"9\":16971.015625,\"10\":14321.8896484375,\"11\":14321.8896484375,\"12\":14321.8896484375,\"13\":16971.015625,\"14\":16971.015625,\"15\":14321.8896484375,\"16\":14321.8896484375,\"17\":14321.8896484375,\"18\":16971.015625,\"19\":14321.8896484375,\"20\":16971.015625,\"21\":16971.015625,\"22\":14321.8896484375,\"23\":16971.015625,\"24\":16971.015625,\"25\":14321.8896484375,\"26\":16971.015625,\"27\":14321.8896484375,\"28\":14321.8896484375,\"29\":16971.015625,\"30\":14321.8896484375,\"31\":14321.8896484375,\"32\":16971.015625,\"33\":16971.015625,\"34\":16971.015625,\"35\":16971.015625,\"36\":14321.8896484375,\"37\":16971.015625,\"38\":14321.8896484375,\"39\":14321.8896484375,\"40\":14321.8896484375,\"41\":16971.015625,\"42\":16971.015625,\"43\":16971.015625,\"44\":16971.015625,\"45\":16971.015625,\"46\":16971.015625,\"47\":16971.015625,\"48\":16971.015625,\"49\":16971.015625,\"50\":14321.8896484375,\"51\":14321.8896484375,\"52\":16971.015625,\"53\":16971.015625,\"54\":14321.8896484375,\"55\":16971.015625,\"56\":14321.8896484375,\"57\":16971.015625,\"58\":16971.015625,\"59\":16971.015625,\"60\":16971.015625,\"61\":14321.8896484375,\"62\":16971.015625,\"63\":14321.8896484375,\"64\":14321.8896484375,\"65\":14321.8896484375,\"66\":16971.015625,\"67\":14321.8896484375,\"68\":16971.015625,\"69\":16971.015625,\"70\":14321.8896484375,\"71\":14321.8896484375,\"72\":14321.8896484375,\"73\":14321.8896484375,\"74\":16971.015625,\"75\":16971.015625,\"76\":16971.015625,\"77\":16971.015625,\"78\":14321.8896484375,\"79\":14321.8896484375,\"80\":14321.8896484375,\"81\":14321.8896484375,\"82\":14321.8896484375,\"83\":14321.8896484375,\"84\":16971.015625,\"85\":16971.015625,\"86\":16971.015625,\"87\":16971.015625,\"88\":14321.8896484375,\"89\":16971.015625,\"90\":16971.015625,\"91\":14321.8896484375,\"92\":14321.8896484375,\"93\":16971.015625,\"94\":16971.015625,\"95\":14321.8896484375,\"96\":16971.015625,\"97\":16971.015625,\"98\":14321.8896484375,\"99\":14321.8896484375,\"100\":16971.015625,\"101\":14321.8896484375,\"102\":14321.8896484375,\"103\":14321.8896484375,\"104\":16971.015625,\"105\":14321.8896484375,\"106\":14321.8896484375,\"107\":16971.015625,\"108\":14321.8896484375,\"109\":14321.8896484375,\"110\":14321.8896484375,\"111\":14321.8896484375,\"112\":14321.8896484375,\"113\":14321.8896484375,\"114\":16971.015625,\"115\":16971.015625,\"116\":16971.015625,\"117\":16971.015625,\"118\":16971.015625,\"119\":16971.015625,\"120\":16971.015625,\"121\":16971.015625,\"122\":16971.015625,\"123\":14321.8896484375,\"124\":16971.015625,\"125\":14321.8896484375,\"126\":16971.015625,\"127\":14321.8896484375,\"128\":14321.8896484375,\"129\":14321.8896484375,\"130\":16971.015625,\"131\":14321.8896484375,\"132\":16971.015625,\"133\":16971.015625,\"134\":14321.8896484375,\"135\":16971.015625,\"136\":16971.015625,\"137\":16971.015625,\"138\":16971.015625,\"139\":16971.015625,\"140\":14321.8896484375,\"141\":14321.8896484375,\"142\":14321.8896484375,\"143\":16971.015625,\"144\":14321.8896484375,\"145\":16971.015625,\"146\":16971.015625,\"147\":16971.015625,\"148\":16971.015625,\"149\":14321.8896484375,\"150\":14321.8896484375,\"151\":16971.015625,\"152\":14321.8896484375,\"153\":16971.015625,\"154\":16971.015625,\"155\":16971.015625,\"156\":16971.015625,\"157\":16971.015625,\"158\":14321.8896484375,\"159\":14321.8896484375,\"160\":14321.8896484375,\"161\":16971.015625,\"162\":14321.8896484375,\"163\":14321.8896484375,\"164\":14321.8896484375,\"165\":16971.015625,\"166\":14321.8896484375,\"167\":14321.8896484375,\"168\":14321.8896484375,\"169\":16971.015625,\"170\":14321.8896484375,\"171\":14321.8896484375,\"172\":16971.015625,\"173\":14321.8896484375,\"174\":16971.015625,\"175\":14321.8896484375,\"176\":16971.015625,\"177\":16971.015625,\"178\":16971.015625,\"179\":16971.015625,\"180\":16971.015625,\"181\":16971.015625,\"182\":16971.015625,\"183\":14321.8896484375,\"184\":14321.8896484375,\"185\":16971.015625,\"186\":16971.015625,\"187\":14321.8896484375,\"188\":16971.015625,\"189\":14321.8896484375,\"190\":16971.015625,\"191\":14321.8896484375,\"192\":14321.8896484375,\"193\":16971.015625,\"194\":16971.015625,\"195\":16971.015625,\"196\":16971.015625,\"197\":14321.8896484375,\"198\":16971.015625,\"199\":16971.015625,\"200\":16971.015625,\"201\":14321.8896484375,\"202\":16971.015625,\"203\":14321.8896484375,\"204\":16971.015625,\"205\":16971.015625,\"206\":16971.015625,\"207\":14321.8896484375,\"208\":14321.8896484375,\"209\":16971.015625,\"210\":16971.015625,\"211\":16971.015625,\"212\":16971.015625,\"213\":16971.015625,\"214\":14321.8896484375,\"215\":16971.015625,\"216\":14321.8896484375,\"217\":16971.015625,\"218\":16971.015625,\"219\":16971.015625,\"220\":16971.015625,\"221\":16971.015625,\"222\":16971.015625,\"223\":16971.015625,\"224\":16971.015625,\"225\":16971.015625,\"226\":14321.8896484375,\"227\":16971.015625,\"228\":16971.015625,\"229\":14321.8896484375,\"230\":16971.015625,\"231\":16971.015625,\"232\":16971.015625,\"233\":16971.015625,\"234\":16971.015625,\"235\":14321.8896484375,\"236\":16971.015625,\"237\":14321.8896484375,\"238\":14321.8896484375,\"239\":16971.015625,\"240\":14321.8896484375,\"241\":16971.015625,\"242\":14321.8896484375,\"243\":14321.8896484375,\"244\":16971.015625,\"245\":16971.015625,\"246\":16971.015625,\"247\":16971.015625,\"248\":16971.015625,\"249\":16971.015625,\"250\":16971.015625,\"251\":16971.015625,\"252\":16971.015625,\"253\":14321.8896484375,\"254\":16971.015625,\"255\":14321.8896484375,\"256\":14321.8896484375,\"257\":14321.8896484375,\"258\":14321.8896484375,\"259\":14321.8896484375,\"260\":14321.8896484375,\"261\":14321.8896484375,\"262\":16971.015625,\"263\":16971.015625,\"264\":14321.8896484375,\"265\":14321.8896484375,\"266\":16971.015625,\"267\":16971.015625,\"268\":14321.8896484375,\"269\":14321.8896484375,\"270\":14321.8896484375,\"271\":16971.015625,\"272\":14321.8896484375,\"273\":14321.8896484375,\"274\":16971.015625,\"275\":16971.015625,\"276\":14321.8896484375,\"277\":14321.8896484375,\"278\":14321.8896484375,\"279\":14321.8896484375,\"280\":16971.015625,\"281\":16971.015625,\"282\":16971.015625,\"283\":16971.015625,\"284\":16971.015625,\"285\":14321.8896484375,\"286\":14321.8896484375,\"287\":14321.8896484375,\"288\":14321.8896484375,\"289\":14321.8896484375,\"290\":14321.8896484375,\"291\":16971.015625,\"292\":16971.015625,\"293\":14321.8896484375,\"294\":14321.8896484375,\"295\":16971.015625,\"296\":16971.015625,\"297\":14321.8896484375,\"298\":16971.015625,\"299\":16971.015625,\"300\":16971.015625,\"301\":16971.015625,\"302\":16971.015625,\"303\":16971.015625,\"304\":16971.015625,\"305\":14321.8896484375,\"306\":14321.8896484375,\"307\":16971.015625,\"308\":14321.8896484375,\"309\":14321.8896484375,\"310\":16971.015625,\"311\":16971.015625,\"312\":16971.015625,\"313\":14321.8896484375,\"314\":16971.015625,\"315\":16971.015625,\"316\":14321.8896484375,\"317\":14321.8896484375,\"318\":14321.8896484375,\"319\":16971.015625,\"320\":16971.015625,\"321\":14321.8896484375,\"322\":16971.015625,\"323\":16971.015625,\"324\":16971.015625,\"325\":14321.8896484375,\"326\":16971.015625,\"327\":14321.8896484375,\"328\":16971.015625,\"329\":14321.8896484375,\"330\":14321.8896484375,\"331\":14321.8896484375,\"332\":16971.015625,\"333\":14321.8896484375,\"334\":16971.015625,\"335\":14321.8896484375,\"336\":16971.015625,\"337\":16971.015625,\"338\":16971.015625,\"339\":14321.8896484375,\"340\":16971.015625,\"341\":16971.015625,\"342\":16971.015625,\"343\":14321.8896484375,\"344\":16971.015625,\"345\":14321.8896484375,\"346\":16971.015625,\"347\":16971.015625,\"348\":14321.8896484375,\"349\":16971.015625,\"350\":16971.015625,\"351\":16971.015625,\"352\":14321.8896484375,\"353\":16971.015625,\"354\":16971.015625,\"355\":14321.8896484375,\"356\":16971.015625,\"357\":14321.8896484375,\"358\":14321.8896484375,\"359\":16971.015625,\"360\":16971.015625,\"361\":14321.8896484375,\"362\":16971.015625,\"363\":14321.8896484375,\"364\":16971.015625,\"365\":14321.8896484375,\"366\":14321.8896484375,\"367\":14321.8896484375,\"368\":14321.8896484375,\"369\":14321.8896484375,\"370\":16971.015625,\"371\":16971.015625,\"372\":16971.015625,\"373\":14321.8896484375,\"374\":14321.8896484375,\"375\":14321.8896484375,\"376\":16971.015625,\"377\":14321.8896484375,\"378\":14321.8896484375,\"379\":16971.015625,\"380\":16971.015625,\"381\":14321.8896484375,\"382\":16971.015625,\"383\":14321.8896484375,\"384\":14321.8896484375,\"385\":16971.015625,\"386\":16971.015625,\"387\":16971.015625,\"388\":16971.015625,\"389\":16971.015625,\"390\":14321.8896484375,\"391\":14321.8896484375,\"392\":14321.8896484375,\"393\":14321.8896484375,\"394\":14321.8896484375,\"395\":14321.8896484375,\"396\":14321.8896484375,\"397\":16971.015625,\"398\":16971.015625,\"399\":14321.8896484375,\"400\":16971.015625,\"401\":16971.015625,\"402\":14321.8896484375,\"403\":16971.015625,\"404\":14321.8896484375,\"405\":16971.015625,\"406\":14321.8896484375,\"407\":14321.8896484375,\"408\":14321.8896484375,\"409\":16971.015625,\"410\":14321.8896484375,\"411\":16971.015625,\"412\":16971.015625,\"413\":16971.015625,\"414\":14321.8896484375,\"415\":14321.8896484375,\"416\":16971.015625,\"417\":16971.015625,\"418\":16971.015625,\"419\":16971.015625,\"420\":16971.015625,\"421\":14321.8896484375,\"422\":14321.8896484375,\"423\":16971.015625,\"424\":14321.8896484375,\"425\":14321.8896484375,\"426\":14321.8896484375,\"427\":16971.015625,\"428\":14321.8896484375,\"429\":16971.015625,\"430\":16971.015625,\"431\":14321.8896484375,\"432\":14321.8896484375,\"433\":14321.8896484375,\"434\":14321.8896484375,\"435\":14321.8896484375,\"436\":16971.015625,\"437\":16971.015625,\"438\":16971.015625,\"439\":14321.8896484375,\"440\":14321.8896484375,\"441\":16971.015625,\"442\":14321.8896484375,\"443\":14321.8896484375,\"444\":16971.015625,\"445\":16971.015625,\"446\":16971.015625,\"447\":16971.015625,\"448\":16971.015625,\"449\":16971.015625,\"450\":14321.8896484375,\"451\":14321.8896484375,\"452\":16971.015625,\"453\":14321.8896484375,\"454\":14321.8896484375,\"455\":14321.8896484375,\"456\":14321.8896484375,\"457\":14321.8896484375,\"458\":16971.015625,\"459\":14321.8896484375,\"460\":16971.015625,\"461\":16971.015625,\"462\":14321.8896484375,\"463\":14321.8896484375,\"464\":16971.015625,\"465\":14321.8896484375,\"466\":14321.8896484375,\"467\":16971.015625,\"468\":16971.015625,\"469\":14321.8896484375,\"470\":16971.015625,\"471\":16971.015625,\"472\":16971.015625,\"473\":14321.8896484375,\"474\":16971.015625,\"475\":16971.015625,\"476\":16971.015625,\"477\":14321.8896484375,\"478\":14321.8896484375,\"479\":16971.015625,\"480\":14321.8896484375,\"481\":16971.015625,\"482\":14321.8896484375,\"483\":16971.015625,\"484\":16971.015625,\"485\":14321.8896484375,\"486\":16971.015625,\"487\":16971.015625,\"488\":14321.8896484375,\"489\":16971.015625,\"490\":14321.8896484375,\"491\":16971.015625,\"492\":16971.015625,\"493\":16971.015625,\"494\":14321.8896484375,\"495\":16971.015625,\"496\":16971.015625,\"497\":16971.015625,\"498\":16971.015625,\"499\":14321.8896484375,\"500\":14321.8896484375,\"501\":14321.8896484375,\"502\":16971.015625,\"503\":14321.8896484375,\"504\":14321.8896484375,\"505\":14321.8896484375,\"506\":16971.015625,\"507\":16971.015625,\"508\":16971.015625,\"509\":14321.8896484375,\"510\":16971.015625,\"511\":14321.8896484375,\"512\":14321.8896484375,\"513\":16971.015625,\"514\":14321.8896484375,\"515\":16971.015625,\"516\":16971.015625,\"517\":14321.8896484375,\"518\":14321.8896484375,\"519\":16971.015625,\"520\":16971.015625,\"521\":14321.8896484375,\"522\":16971.015625,\"523\":14321.8896484375,\"524\":14321.8896484375,\"525\":16971.015625,\"526\":16971.015625,\"527\":14321.8896484375,\"528\":16971.015625,\"529\":14321.8896484375,\"530\":16971.015625,\"531\":16971.015625,\"532\":14321.8896484375,\"533\":16971.015625,\"534\":16971.015625,\"535\":16971.015625,\"536\":16971.015625,\"537\":16971.015625,\"538\":14321.8896484375,\"539\":14321.8896484375,\"540\":16971.015625,\"541\":16971.015625,\"542\":14321.8896484375,\"543\":16971.015625,\"544\":14321.8896484375,\"545\":16971.015625,\"546\":16971.015625,\"547\":16971.015625,\"548\":16971.015625,\"549\":16971.015625,\"550\":14321.8896484375,\"551\":16971.015625,\"552\":16971.015625,\"553\":16971.015625,\"554\":14321.8896484375,\"555\":14321.8896484375,\"556\":14321.8896484375,\"557\":14321.8896484375,\"558\":16971.015625,\"559\":16971.015625,\"560\":16971.015625,\"561\":16971.015625,\"562\":14321.8896484375,\"563\":14321.8896484375,\"564\":16971.015625,\"565\":16971.015625,\"566\":16971.015625,\"567\":14321.8896484375,\"568\":14321.8896484375,\"569\":16971.015625,\"570\":16971.015625,\"571\":14321.8896484375,\"572\":16971.015625,\"573\":14321.8896484375,\"574\":14321.8896484375,\"575\":16971.015625,\"576\":14321.8896484375,\"577\":14321.8896484375,\"578\":14321.8896484375,\"579\":16971.015625,\"580\":16971.015625,\"581\":16971.015625,\"582\":16971.015625,\"583\":16971.015625,\"584\":14321.8896484375,\"585\":16971.015625,\"586\":16971.015625,\"587\":16971.015625,\"588\":14321.8896484375,\"589\":14321.8896484375,\"590\":14321.8896484375,\"591\":16971.015625,\"592\":16971.015625,\"593\":16971.015625,\"594\":14321.8896484375,\"595\":14321.8896484375,\"596\":16971.015625,\"597\":14321.8896484375,\"598\":14321.8896484375,\"599\":16971.015625,\"600\":16971.015625,\"601\":16971.015625,\"602\":14321.8896484375,\"603\":16971.015625,\"604\":16971.015625,\"605\":16971.015625,\"606\":16971.015625,\"607\":16971.015625,\"608\":16971.015625,\"609\":16971.015625,\"610\":16971.015625,\"611\":16971.015625,\"612\":14321.8896484375,\"613\":14321.8896484375,\"614\":14321.8896484375,\"615\":16971.015625,\"616\":16971.015625,\"617\":14321.8896484375,\"618\":16971.015625,\"619\":14321.8896484375,\"620\":14321.8896484375,\"621\":14321.8896484375,\"622\":14321.8896484375,\"623\":14321.8896484375,\"624\":16971.015625,\"625\":16971.015625,\"626\":16971.015625,\"627\":14321.8896484375,\"628\":16971.015625,\"629\":16971.015625,\"630\":16971.015625,\"631\":16971.015625,\"632\":14321.8896484375,\"633\":16971.015625,\"634\":14321.8896484375,\"635\":16971.015625,\"636\":16971.015625,\"637\":14321.8896484375,\"638\":16971.015625,\"639\":16971.015625,\"640\":14321.8896484375,\"641\":14321.8896484375,\"642\":16971.015625,\"643\":16971.015625,\"644\":16971.015625,\"645\":14321.8896484375,\"646\":16971.015625,\"647\":14321.8896484375,\"648\":16971.015625,\"649\":14321.8896484375,\"650\":16971.015625,\"651\":16971.015625,\"652\":16971.015625,\"653\":14321.8896484375,\"654\":14321.8896484375,\"655\":16971.015625,\"656\":14321.8896484375,\"657\":14321.8896484375,\"658\":14321.8896484375,\"659\":14321.8896484375,\"660\":14321.8896484375,\"661\":16971.015625,\"662\":14321.8896484375,\"663\":16971.015625,\"664\":16971.015625,\"665\":14321.8896484375,\"666\":14321.8896484375,\"667\":16971.015625,\"668\":14321.8896484375,\"669\":16971.015625,\"670\":14321.8896484375,\"671\":14321.8896484375,\"672\":14321.8896484375,\"673\":14321.8896484375,\"674\":16971.015625,\"675\":16971.015625,\"676\":16971.015625,\"677\":16971.015625,\"678\":16971.015625,\"679\":16971.015625,\"680\":14321.8896484375,\"681\":14321.8896484375,\"682\":16971.015625,\"683\":16971.015625,\"684\":16971.015625,\"685\":14321.8896484375,\"686\":14321.8896484375,\"687\":16971.015625,\"688\":14321.8896484375,\"689\":16971.015625,\"690\":16971.015625,\"691\":16971.015625,\"692\":14321.8896484375,\"693\":14321.8896484375,\"694\":16971.015625,\"695\":14321.8896484375,\"696\":16971.015625,\"697\":16971.015625,\"698\":14321.8896484375,\"699\":14321.8896484375,\"700\":16971.015625,\"701\":14321.8896484375,\"702\":14321.8896484375,\"703\":14321.8896484375,\"704\":14321.8896484375,\"705\":16971.015625,\"706\":14321.8896484375,\"707\":14321.8896484375,\"708\":16971.015625,\"709\":16971.015625,\"710\":16971.015625,\"711\":14321.8896484375,\"712\":14321.8896484375,\"713\":16971.015625,\"714\":14321.8896484375,\"715\":16971.015625,\"716\":16971.015625,\"717\":14321.8896484375,\"718\":14321.8896484375,\"719\":16971.015625,\"720\":14321.8896484375,\"721\":16971.015625,\"722\":14321.8896484375,\"723\":16971.015625,\"724\":14321.8896484375,\"725\":16971.015625,\"726\":14321.8896484375,\"727\":16971.015625,\"728\":16971.015625,\"729\":14321.8896484375,\"730\":16971.015625,\"731\":16971.015625,\"732\":16971.015625,\"733\":16971.015625,\"734\":16971.015625,\"735\":14321.8896484375,\"736\":16971.015625,\"737\":16971.015625,\"738\":14321.8896484375,\"739\":14321.8896484375,\"740\":16971.015625,\"741\":16971.015625,\"742\":14321.8896484375,\"743\":14321.8896484375,\"744\":14321.8896484375,\"745\":16971.015625,\"746\":16971.015625,\"747\":14321.8896484375,\"748\":14321.8896484375,\"749\":14321.8896484375,\"750\":16971.015625,\"751\":14321.8896484375,\"752\":14321.8896484375,\"753\":16971.015625,\"754\":16971.015625,\"755\":16971.015625,\"756\":16971.015625,\"757\":14321.8896484375,\"758\":16971.015625,\"759\":16971.015625,\"760\":16971.015625,\"761\":16971.015625,\"762\":14321.8896484375,\"763\":14321.8896484375,\"764\":14321.8896484375,\"765\":16971.015625,\"766\":14321.8896484375,\"767\":14321.8896484375,\"768\":16971.015625,\"769\":14321.8896484375,\"770\":14321.8896484375,\"771\":16971.015625,\"772\":16971.015625,\"773\":14321.8896484375,\"774\":16971.015625,\"775\":16971.015625,\"776\":16971.015625,\"777\":14321.8896484375,\"778\":16971.015625,\"779\":14321.8896484375,\"780\":16971.015625,\"781\":16971.015625,\"782\":14321.8896484375,\"783\":14321.8896484375,\"784\":14321.8896484375,\"785\":14321.8896484375,\"786\":14321.8896484375,\"787\":16971.015625,\"788\":16971.015625,\"789\":14321.8896484375,\"790\":16971.015625,\"791\":14321.8896484375,\"792\":16971.015625,\"793\":16971.015625,\"794\":16971.015625,\"795\":16971.015625,\"796\":16971.015625,\"797\":14321.8896484375,\"798\":16971.015625,\"799\":16971.015625,\"800\":16971.015625,\"801\":16971.015625,\"802\":16971.015625,\"803\":16971.015625,\"804\":16971.015625,\"805\":16971.015625,\"806\":14321.8896484375,\"807\":14321.8896484375,\"808\":14321.8896484375,\"809\":14321.8896484375,\"810\":14321.8896484375,\"811\":14321.8896484375,\"812\":16971.015625,\"813\":16971.015625,\"814\":14321.8896484375,\"815\":14321.8896484375,\"816\":16971.015625,\"817\":16971.015625,\"818\":16971.015625,\"819\":16971.015625,\"820\":16971.015625,\"821\":14321.8896484375,\"822\":14321.8896484375,\"823\":16971.015625,\"824\":16971.015625,\"825\":16971.015625,\"826\":16971.015625,\"827\":16971.015625,\"828\":16971.015625,\"829\":16971.015625,\"830\":14321.8896484375,\"831\":16971.015625,\"832\":14321.8896484375,\"833\":16971.015625,\"834\":14321.8896484375,\"835\":14321.8896484375,\"836\":16971.015625,\"837\":16971.015625,\"838\":16971.015625,\"839\":16971.015625,\"840\":16971.015625,\"841\":14321.8896484375,\"842\":16971.015625,\"843\":16971.015625,\"844\":14321.8896484375,\"845\":14321.8896484375,\"846\":16971.015625,\"847\":16971.015625,\"848\":16971.015625,\"849\":14321.8896484375,\"850\":16971.015625,\"851\":16971.015625,\"852\":14321.8896484375,\"853\":16971.015625,\"854\":14321.8896484375,\"855\":14321.8896484375,\"856\":16971.015625,\"857\":16971.015625,\"858\":14321.8896484375,\"859\":14321.8896484375,\"860\":14321.8896484375,\"861\":16971.015625,\"862\":14321.8896484375,\"863\":16971.015625,\"864\":14321.8896484375,\"865\":14321.8896484375,\"866\":14321.8896484375,\"867\":16971.015625,\"868\":16971.015625,\"869\":14321.8896484375,\"870\":16971.015625,\"871\":16971.015625,\"872\":14321.8896484375,\"873\":16971.015625,\"874\":14321.8896484375,\"875\":14321.8896484375,\"876\":14321.8896484375,\"877\":14321.8896484375,\"878\":14321.8896484375,\"879\":16971.015625,\"880\":14321.8896484375,\"881\":16971.015625,\"882\":16971.015625,\"883\":16971.015625,\"884\":16971.015625,\"885\":16971.015625,\"886\":14321.8896484375,\"887\":16971.015625,\"888\":16971.015625,\"889\":14321.8896484375,\"890\":14321.8896484375,\"891\":16971.015625,\"892\":14321.8896484375,\"893\":16971.015625,\"894\":16971.015625,\"895\":14321.8896484375,\"896\":16971.015625,\"897\":16971.015625,\"898\":16971.015625,\"899\":16971.015625,\"900\":16971.015625,\"901\":16971.015625,\"902\":16971.015625,\"903\":14321.8896484375,\"904\":14321.8896484375,\"905\":14321.8896484375,\"906\":14321.8896484375,\"907\":16971.015625,\"908\":16971.015625,\"909\":16971.015625,\"910\":14321.8896484375,\"911\":14321.8896484375,\"912\":16971.015625,\"913\":16971.015625,\"914\":16971.015625,\"915\":16971.015625,\"916\":14321.8896484375,\"917\":14321.8896484375,\"918\":14321.8896484375,\"919\":14321.8896484375,\"920\":16971.015625,\"921\":16971.015625,\"922\":16971.015625,\"923\":16971.015625,\"924\":16971.015625,\"925\":16971.015625,\"926\":14321.8896484375,\"927\":16971.015625,\"928\":16971.015625,\"929\":16971.015625,\"930\":16971.015625,\"931\":16971.015625,\"932\":14321.8896484375,\"933\":14321.8896484375,\"934\":16971.015625,\"935\":16971.015625,\"936\":14321.8896484375,\"937\":16971.015625,\"938\":16971.015625,\"939\":14321.8896484375,\"940\":14321.8896484375,\"941\":14321.8896484375,\"942\":16971.015625,\"943\":16971.015625,\"944\":14321.8896484375,\"945\":14321.8896484375,\"946\":16971.015625,\"947\":16971.015625,\"948\":16971.015625,\"949\":16971.015625,\"950\":16971.015625,\"951\":14321.8896484375,\"952\":16971.015625,\"953\":14321.8896484375,\"954\":14321.8896484375,\"955\":14321.8896484375,\"956\":16971.015625,\"957\":16971.015625,\"958\":16971.015625,\"959\":16971.015625,\"960\":14321.8896484375,\"961\":14321.8896484375,\"962\":16971.015625,\"963\":16971.015625,\"964\":16971.015625,\"965\":14321.8896484375,\"966\":16971.015625,\"967\":14321.8896484375,\"968\":14321.8896484375,\"969\":16971.015625,\"970\":16971.015625,\"971\":16971.015625,\"972\":16971.015625,\"973\":16971.015625,\"974\":14321.8896484375,\"975\":16971.015625,\"976\":16971.015625,\"977\":14321.8896484375,\"978\":16971.015625,\"979\":16971.015625,\"980\":14321.8896484375,\"981\":14321.8896484375,\"982\":16971.015625,\"983\":14321.8896484375,\"984\":14321.8896484375,\"985\":16971.015625,\"986\":16971.015625,\"987\":16971.015625,\"988\":14321.8896484375,\"989\":16971.015625,\"990\":14321.8896484375,\"991\":16971.015625,\"992\":16971.015625,\"993\":16971.015625,\"994\":14321.8896484375,\"995\":14321.8896484375,\"996\":16971.015625,\"997\":14321.8896484375,\"998\":16971.015625,\"999\":16971.015625,\"1000\":16971.015625,\"1001\":16971.015625,\"1002\":14321.8896484375,\"1003\":16971.015625,\"1004\":16971.015625,\"1005\":16971.015625,\"1006\":16971.015625,\"1007\":16971.015625,\"1008\":16971.015625,\"1009\":16971.015625,\"1010\":14321.8896484375,\"1011\":16971.015625,\"1012\":16971.015625,\"1013\":14321.8896484375,\"1014\":14321.8896484375,\"1015\":16971.015625,\"1016\":16971.015625,\"1017\":16971.015625,\"1018\":16971.015625,\"1019\":14321.8896484375,\"1020\":16971.015625,\"1021\":16971.015625,\"1022\":16971.015625,\"1023\":16971.015625,\"1024\":14321.8896484375,\"1025\":14321.8896484375,\"1026\":14321.8896484375,\"1027\":16971.015625,\"1028\":14321.8896484375,\"1029\":16971.015625,\"1030\":14321.8896484375,\"1031\":14321.8896484375,\"1032\":14321.8896484375,\"1033\":16971.015625,\"1034\":16971.015625,\"1035\":16971.015625,\"1036\":14321.8896484375,\"1037\":14321.8896484375,\"1038\":16971.015625,\"1039\":16971.015625,\"1040\":16971.015625,\"1041\":14321.8896484375,\"1042\":14321.8896484375,\"1043\":14321.8896484375,\"1044\":14321.8896484375,\"1045\":16971.015625,\"1046\":14321.8896484375,\"1047\":14321.8896484375,\"1048\":16971.015625,\"1049\":16971.015625,\"1050\":14321.8896484375,\"1051\":14321.8896484375,\"1052\":16971.015625,\"1053\":16971.015625,\"1054\":16971.015625,\"1055\":16971.015625,\"1056\":14321.8896484375,\"1057\":14321.8896484375,\"1058\":16971.015625,\"1059\":16971.015625,\"1060\":14321.8896484375,\"1061\":16971.015625,\"1062\":16971.015625,\"1063\":14321.8896484375,\"1064\":14321.8896484375,\"1065\":14321.8896484375,\"1066\":14321.8896484375,\"1067\":16971.015625,\"1068\":16971.015625,\"1069\":14321.8896484375,\"1070\":14321.8896484375,\"1071\":16971.015625,\"1072\":14321.8896484375,\"1073\":14321.8896484375,\"1074\":16971.015625,\"1075\":16971.015625,\"1076\":16971.015625,\"1077\":14321.8896484375,\"1078\":16971.015625,\"1079\":14321.8896484375,\"1080\":16971.015625,\"1081\":14321.8896484375,\"1082\":16971.015625,\"1083\":16971.015625,\"1084\":16971.015625,\"1085\":16971.015625,\"1086\":16971.015625,\"1087\":14321.8896484375,\"1088\":14321.8896484375,\"1089\":16971.015625,\"1090\":16971.015625,\"1091\":16971.015625,\"1092\":14321.8896484375,\"1093\":16971.015625,\"1094\":16971.015625,\"1095\":16971.015625,\"1096\":14321.8896484375,\"1097\":16971.015625,\"1098\":16971.015625,\"1099\":16971.015625,\"1100\":14321.8896484375,\"1101\":16971.015625,\"1102\":14321.8896484375,\"1103\":14321.8896484375,\"1104\":16971.015625,\"1105\":16971.015625,\"1106\":16971.015625,\"1107\":16971.015625,\"1108\":14321.8896484375,\"1109\":16971.015625,\"1110\":16971.015625,\"1111\":14321.8896484375,\"1112\":14321.8896484375,\"1113\":14321.8896484375,\"1114\":14321.8896484375,\"1115\":14321.8896484375,\"1116\":16971.015625,\"1117\":16971.015625,\"1118\":14321.8896484375,\"1119\":14321.8896484375,\"1120\":14321.8896484375,\"1121\":16971.015625,\"1122\":16971.015625,\"1123\":16971.015625,\"1124\":14321.8896484375,\"1125\":14321.8896484375,\"1126\":16971.015625,\"1127\":16971.015625,\"1128\":16971.015625,\"1129\":14321.8896484375,\"1130\":16971.015625,\"1131\":16971.015625,\"1132\":14321.8896484375,\"1133\":16971.015625,\"1134\":16971.015625,\"1135\":16971.015625,\"1136\":14321.8896484375,\"1137\":16971.015625,\"1138\":14321.8896484375,\"1139\":16971.015625,\"1140\":16971.015625,\"1141\":14321.8896484375,\"1142\":14321.8896484375,\"1143\":16971.015625,\"1144\":16971.015625,\"1145\":14321.8896484375,\"1146\":16971.015625,\"1147\":16971.015625,\"1148\":14321.8896484375,\"1149\":16971.015625,\"1150\":14321.8896484375,\"1151\":14321.8896484375,\"1152\":14321.8896484375,\"1153\":16971.015625,\"1154\":16971.015625,\"1155\":14321.8896484375,\"1156\":14321.8896484375,\"1157\":14321.8896484375,\"1158\":16971.015625,\"1159\":16971.015625,\"1160\":16971.015625,\"1161\":14321.8896484375,\"1162\":14321.8896484375,\"1163\":14321.8896484375,\"1164\":16971.015625,\"1165\":16971.015625,\"1166\":14321.8896484375,\"1167\":14321.8896484375,\"1168\":14321.8896484375,\"1169\":16971.015625,\"1170\":14321.8896484375,\"1171\":16971.015625,\"1172\":16971.015625,\"1173\":14321.8896484375,\"1174\":16971.015625,\"1175\":16971.015625,\"1176\":14321.8896484375,\"1177\":16971.015625,\"1178\":14321.8896484375,\"1179\":14321.8896484375,\"1180\":16971.015625,\"1181\":14321.8896484375,\"1182\":14321.8896484375,\"1183\":16971.015625,\"1184\":16971.015625,\"1185\":16971.015625,\"1186\":16971.015625,\"1187\":14321.8896484375,\"1188\":16971.015625,\"1189\":14321.8896484375,\"1190\":14321.8896484375,\"1191\":14321.8896484375,\"1192\":16971.015625,\"1193\":16971.015625,\"1194\":16971.015625,\"1195\":16971.015625,\"1196\":16971.015625,\"1197\":16971.015625,\"1198\":16971.015625,\"1199\":16971.015625,\"1200\":16971.015625,\"1201\":14321.8896484375,\"1202\":14321.8896484375,\"1203\":16971.015625,\"1204\":16971.015625,\"1205\":14321.8896484375,\"1206\":16971.015625,\"1207\":14321.8896484375,\"1208\":16971.015625,\"1209\":16971.015625,\"1210\":16971.015625,\"1211\":16971.015625,\"1212\":14321.8896484375,\"1213\":16971.015625,\"1214\":14321.8896484375,\"1215\":14321.8896484375,\"1216\":14321.8896484375,\"1217\":16971.015625,\"1218\":14321.8896484375,\"1219\":16971.015625,\"1220\":16971.015625,\"1221\":14321.8896484375,\"1222\":14321.8896484375,\"1223\":14321.8896484375,\"1224\":14321.8896484375,\"1225\":16971.015625,\"1226\":16971.015625,\"1227\":16971.015625,\"1228\":16971.015625,\"1229\":14321.8896484375,\"1230\":14321.8896484375,\"1231\":14321.8896484375,\"1232\":14321.8896484375,\"1233\":14321.8896484375,\"1234\":14321.8896484375,\"1235\":16971.015625,\"1236\":16971.015625,\"1237\":16971.015625,\"1238\":16971.015625,\"1239\":14321.8896484375,\"1240\":16971.015625,\"1241\":16971.015625,\"1242\":14321.8896484375,\"1243\":14321.8896484375,\"1244\":16971.015625,\"1245\":16971.015625,\"1246\":14321.8896484375,\"1247\":16971.015625,\"1248\":16971.015625,\"1249\":14321.8896484375,\"1250\":14321.8896484375,\"1251\":16971.015625,\"1252\":14321.8896484375,\"1253\":14321.8896484375,\"1254\":14321.8896484375,\"1255\":16971.015625,\"1256\":14321.8896484375,\"1257\":14321.8896484375,\"1258\":16971.015625,\"1259\":14321.8896484375,\"1260\":14321.8896484375,\"1261\":14321.8896484375,\"1262\":14321.8896484375,\"1263\":14321.8896484375,\"1264\":14321.8896484375,\"1265\":16971.015625,\"1266\":16971.015625,\"1267\":16971.015625,\"1268\":16971.015625,\"1269\":16971.015625,\"1270\":16971.015625,\"1271\":16971.015625,\"1272\":16971.015625,\"1273\":16971.015625,\"1274\":14321.8896484375,\"1275\":16971.015625,\"1276\":14321.8896484375,\"1277\":16971.015625,\"1278\":14321.8896484375,\"1279\":14321.8896484375,\"1280\":14321.8896484375,\"1281\":16971.015625,\"1282\":14321.8896484375,\"1283\":16971.015625,\"1284\":16971.015625,\"1285\":14321.8896484375,\"1286\":16971.015625,\"1287\":16971.015625,\"1288\":16971.015625,\"1289\":16971.015625,\"1290\":16971.015625,\"1291\":14321.8896484375,\"1292\":14321.8896484375,\"1293\":14321.8896484375,\"1294\":16971.015625,\"1295\":14321.8896484375,\"1296\":16971.015625,\"1297\":16971.015625,\"1298\":16971.015625,\"1299\":16971.015625,\"1300\":14321.8896484375,\"1301\":14321.8896484375,\"1302\":16971.015625,\"1303\":14321.8896484375,\"1304\":16971.015625,\"1305\":16971.015625,\"1306\":16971.015625,\"1307\":16971.015625,\"1308\":16971.015625,\"1309\":14321.8896484375,\"1310\":14321.8896484375,\"1311\":14321.8896484375,\"1312\":16971.015625,\"1313\":14321.8896484375,\"1314\":14321.8896484375,\"1315\":14321.8896484375,\"1316\":16971.015625,\"1317\":14321.8896484375,\"1318\":14321.8896484375,\"1319\":14321.8896484375,\"1320\":16971.015625,\"1321\":14321.8896484375,\"1322\":14321.8896484375,\"1323\":16971.015625,\"1324\":14321.8896484375,\"1325\":16971.015625,\"1326\":14321.8896484375,\"1327\":16971.015625,\"1328\":16971.015625,\"1329\":16971.015625,\"1330\":16971.015625,\"1331\":16971.015625,\"1332\":16971.015625,\"1333\":16971.015625,\"1334\":14321.8896484375,\"1335\":14321.8896484375,\"1336\":16971.015625,\"1337\":16971.015625,\"1338\":14321.8896484375,\"1339\":16971.015625,\"1340\":14321.8896484375,\"1341\":16971.015625,\"1342\":14321.8896484375,\"1343\":14321.8896484375,\"1344\":16971.015625,\"1345\":16971.015625,\"1346\":16971.015625,\"1347\":16971.015625,\"1348\":14321.8896484375,\"1349\":16971.015625,\"1350\":16971.015625,\"1351\":16971.015625,\"1352\":14321.8896484375,\"1353\":16971.015625,\"1354\":14321.8896484375,\"1355\":16971.015625,\"1356\":16971.015625,\"1357\":16971.015625,\"1358\":14321.8896484375,\"1359\":14321.8896484375,\"1360\":16971.015625,\"1361\":16971.015625,\"1362\":16971.015625,\"1363\":16971.015625,\"1364\":16971.015625,\"1365\":14321.8896484375,\"1366\":16971.015625,\"1367\":14321.8896484375,\"1368\":16971.015625,\"1369\":16971.015625,\"1370\":16971.015625,\"1371\":16971.015625,\"1372\":16971.015625,\"1373\":16971.015625,\"1374\":16971.015625,\"1375\":16971.015625,\"1376\":16971.015625,\"1377\":14321.8896484375,\"1378\":16971.015625,\"1379\":16971.015625,\"1380\":14321.8896484375,\"1381\":16971.015625,\"1382\":16971.015625,\"1383\":16971.015625,\"1384\":16971.015625,\"1385\":16971.015625,\"1386\":14321.8896484375,\"1387\":16971.015625,\"1388\":14321.8896484375,\"1389\":14321.8896484375,\"1390\":16971.015625,\"1391\":14321.8896484375,\"1392\":16971.015625,\"1393\":14321.8896484375,\"1394\":14321.8896484375,\"1395\":16971.015625,\"1396\":16971.015625,\"1397\":16971.015625,\"1398\":16971.015625,\"1399\":16971.015625,\"1400\":16971.015625,\"1401\":16971.015625,\"1402\":16971.015625,\"1403\":16971.015625,\"1404\":14321.8896484375,\"1405\":16971.015625,\"1406\":14321.8896484375,\"1407\":14321.8896484375,\"1408\":14321.8896484375,\"1409\":14321.8896484375,\"1410\":14321.8896484375,\"1411\":14321.8896484375,\"1412\":14321.8896484375,\"1413\":16971.015625,\"1414\":16971.015625,\"1415\":14321.8896484375,\"1416\":14321.8896484375,\"1417\":16971.015625,\"1418\":16971.015625,\"1419\":14321.8896484375,\"1420\":14321.8896484375,\"1421\":14321.8896484375,\"1422\":16971.015625,\"1423\":14321.8896484375,\"1424\":14321.8896484375,\"1425\":16971.015625,\"1426\":16971.015625,\"1427\":14321.8896484375,\"1428\":14321.8896484375,\"1429\":14321.8896484375,\"1430\":14321.8896484375,\"1431\":16971.015625,\"1432\":16971.015625,\"1433\":16971.015625,\"1434\":16971.015625,\"1435\":16971.015625,\"1436\":14321.8896484375,\"1437\":14321.8896484375,\"1438\":14321.8896484375,\"1439\":14321.8896484375,\"1440\":14321.8896484375,\"1441\":14321.8896484375,\"1442\":16971.015625,\"1443\":16971.015625,\"1444\":14321.8896484375,\"1445\":14321.8896484375,\"1446\":16971.015625,\"1447\":16971.015625,\"1448\":14321.8896484375,\"1449\":16971.015625,\"1450\":16971.015625,\"1451\":16971.015625,\"1452\":16971.015625,\"1453\":16971.015625,\"1454\":16971.015625,\"1455\":16971.015625,\"1456\":14321.8896484375,\"1457\":14321.8896484375,\"1458\":16971.015625,\"1459\":14321.8896484375,\"1460\":14321.8896484375,\"1461\":16971.015625,\"1462\":16971.015625,\"1463\":16971.015625,\"1464\":14321.8896484375,\"1465\":16971.015625,\"1466\":16971.015625,\"1467\":14321.8896484375,\"1468\":14321.8896484375,\"1469\":14321.8896484375,\"1470\":16971.015625,\"1471\":16971.015625,\"1472\":14321.8896484375,\"1473\":16971.015625,\"1474\":16971.015625,\"1475\":16971.015625,\"1476\":14321.8896484375,\"1477\":16971.015625,\"1478\":14321.8896484375,\"1479\":16971.015625,\"1480\":14321.8896484375,\"1481\":14321.8896484375,\"1482\":14321.8896484375,\"1483\":16971.015625,\"1484\":14321.8896484375,\"1485\":16971.015625,\"1486\":14321.8896484375,\"1487\":16971.015625,\"1488\":16971.015625,\"1489\":16971.015625,\"1490\":14321.8896484375,\"1491\":16971.015625,\"1492\":16971.015625,\"1493\":16971.015625,\"1494\":14321.8896484375,\"1495\":16971.015625,\"1496\":14321.8896484375,\"1497\":16971.015625,\"1498\":16971.015625,\"1499\":14321.8896484375,\"1500\":16971.015625,\"1501\":16971.015625,\"1502\":16971.015625,\"1503\":14321.8896484375,\"1504\":16971.015625,\"1505\":16971.015625,\"1506\":14321.8896484375,\"1507\":16971.015625,\"1508\":14321.8896484375,\"1509\":14321.8896484375,\"1510\":16971.015625,\"1511\":16971.015625,\"1512\":14321.8896484375,\"1513\":16971.015625,\"1514\":14321.8896484375,\"1515\":16971.015625,\"1516\":14321.8896484375,\"1517\":14321.8896484375,\"1518\":14321.8896484375,\"1519\":14321.8896484375,\"1520\":14321.8896484375,\"1521\":16971.015625,\"1522\":16971.015625,\"1523\":16971.015625,\"1524\":14321.8896484375,\"1525\":14321.8896484375,\"1526\":14321.8896484375,\"1527\":16971.015625,\"1528\":14321.8896484375,\"1529\":14321.8896484375,\"1530\":16971.015625,\"1531\":16971.015625,\"1532\":14321.8896484375,\"1533\":16971.015625,\"1534\":14321.8896484375,\"1535\":14321.8896484375,\"1536\":16971.015625,\"1537\":16971.015625,\"1538\":16971.015625,\"1539\":16971.015625,\"1540\":16971.015625,\"1541\":14321.8896484375,\"1542\":14321.8896484375,\"1543\":14321.8896484375,\"1544\":14321.8896484375,\"1545\":14321.8896484375,\"1546\":14321.8896484375,\"1547\":14321.8896484375,\"1548\":16971.015625,\"1549\":16971.015625,\"1550\":14321.8896484375,\"1551\":16971.015625,\"1552\":16971.015625,\"1553\":14321.8896484375,\"1554\":16971.015625,\"1555\":14321.8896484375,\"1556\":16971.015625,\"1557\":14321.8896484375,\"1558\":14321.8896484375,\"1559\":14321.8896484375,\"1560\":16971.015625,\"1561\":14321.8896484375,\"1562\":16971.015625,\"1563\":16971.015625,\"1564\":16971.015625,\"1565\":14321.8896484375,\"1566\":14321.8896484375,\"1567\":16971.015625,\"1568\":16971.015625,\"1569\":16971.015625,\"1570\":16971.015625,\"1571\":16971.015625,\"1572\":14321.8896484375,\"1573\":14321.8896484375,\"1574\":16971.015625,\"1575\":14321.8896484375,\"1576\":14321.8896484375,\"1577\":14321.8896484375,\"1578\":16971.015625,\"1579\":14321.8896484375,\"1580\":16971.015625,\"1581\":16971.015625,\"1582\":14321.8896484375,\"1583\":14321.8896484375,\"1584\":14321.8896484375,\"1585\":14321.8896484375,\"1586\":14321.8896484375,\"1587\":16971.015625,\"1588\":16971.015625,\"1589\":16971.015625,\"1590\":14321.8896484375,\"1591\":14321.8896484375,\"1592\":16971.015625,\"1593\":14321.8896484375,\"1594\":14321.8896484375,\"1595\":16971.015625,\"1596\":16971.015625,\"1597\":16971.015625,\"1598\":16971.015625,\"1599\":16971.015625,\"1600\":16971.015625,\"1601\":14321.8896484375,\"1602\":14321.8896484375,\"1603\":16971.015625,\"1604\":14321.8896484375,\"1605\":14321.8896484375,\"1606\":14321.8896484375,\"1607\":14321.8896484375,\"1608\":14321.8896484375,\"1609\":16971.015625,\"1610\":14321.8896484375,\"1611\":16971.015625,\"1612\":16971.015625,\"1613\":14321.8896484375,\"1614\":14321.8896484375,\"1615\":16971.015625,\"1616\":14321.8896484375,\"1617\":14321.8896484375,\"1618\":16971.015625,\"1619\":16971.015625,\"1620\":14321.8896484375,\"1621\":16971.015625,\"1622\":16971.015625,\"1623\":16971.015625,\"1624\":14321.8896484375,\"1625\":16971.015625,\"1626\":16971.015625,\"1627\":16971.015625,\"1628\":14321.8896484375,\"1629\":14321.8896484375,\"1630\":16971.015625,\"1631\":14321.8896484375,\"1632\":16971.015625,\"1633\":14321.8896484375,\"1634\":16971.015625,\"1635\":16971.015625,\"1636\":14321.8896484375,\"1637\":16971.015625,\"1638\":16971.015625,\"1639\":14321.8896484375,\"1640\":16971.015625,\"1641\":14321.8896484375,\"1642\":16971.015625,\"1643\":16971.015625,\"1644\":16971.015625,\"1645\":14321.8896484375,\"1646\":16971.015625,\"1647\":16971.015625,\"1648\":16971.015625,\"1649\":16971.015625,\"1650\":14321.8896484375,\"1651\":14321.8896484375,\"1652\":14321.8896484375,\"1653\":16971.015625,\"1654\":14321.8896484375,\"1655\":14321.8896484375,\"1656\":14321.8896484375,\"1657\":16971.015625,\"1658\":16971.015625,\"1659\":16971.015625,\"1660\":14321.8896484375,\"1661\":16971.015625,\"1662\":14321.8896484375,\"1663\":14321.8896484375,\"1664\":16971.015625,\"1665\":14321.8896484375,\"1666\":16971.015625,\"1667\":16971.015625,\"1668\":14321.8896484375,\"1669\":14321.8896484375,\"1670\":16971.015625,\"1671\":16971.015625,\"1672\":14321.8896484375,\"1673\":16971.015625,\"1674\":14321.8896484375,\"1675\":14321.8896484375,\"1676\":16971.015625,\"1677\":16971.015625,\"1678\":14321.8896484375,\"1679\":16971.015625,\"1680\":14321.8896484375,\"1681\":16971.015625,\"1682\":16971.015625,\"1683\":14321.8896484375,\"1684\":16971.015625,\"1685\":16971.015625,\"1686\":16971.015625,\"1687\":16971.015625,\"1688\":16971.015625,\"1689\":14321.8896484375,\"1690\":14321.8896484375,\"1691\":16971.015625,\"1692\":16971.015625,\"1693\":14321.8896484375,\"1694\":16971.015625,\"1695\":14321.8896484375,\"1696\":16971.015625,\"1697\":16971.015625,\"1698\":16971.015625,\"1699\":16971.015625,\"1700\":16971.015625,\"1701\":14321.8896484375,\"1702\":16971.015625,\"1703\":16971.015625,\"1704\":16971.015625,\"1705\":14321.8896484375,\"1706\":14321.8896484375,\"1707\":14321.8896484375,\"1708\":14321.8896484375,\"1709\":16971.015625,\"1710\":16971.015625,\"1711\":16971.015625,\"1712\":16971.015625,\"1713\":14321.8896484375,\"1714\":14321.8896484375,\"1715\":16971.015625,\"1716\":16971.015625,\"1717\":16971.015625,\"1718\":14321.8896484375,\"1719\":14321.8896484375,\"1720\":16971.015625,\"1721\":16971.015625,\"1722\":14321.8896484375,\"1723\":16971.015625,\"1724\":14321.8896484375,\"1725\":14321.8896484375,\"1726\":16971.015625,\"1727\":14321.8896484375,\"1728\":14321.8896484375,\"1729\":14321.8896484375,\"1730\":16971.015625,\"1731\":16971.015625,\"1732\":16971.015625,\"1733\":16971.015625,\"1734\":16971.015625,\"1735\":14321.8896484375,\"1736\":16971.015625,\"1737\":16971.015625,\"1738\":16971.015625,\"1739\":14321.8896484375,\"1740\":14321.8896484375,\"1741\":14321.8896484375,\"1742\":16971.015625,\"1743\":16971.015625,\"1744\":16971.015625,\"1745\":14321.8896484375,\"1746\":14321.8896484375,\"1747\":16971.015625,\"1748\":14321.8896484375,\"1749\":14321.8896484375,\"1750\":16971.015625,\"1751\":16971.015625,\"1752\":16971.015625,\"1753\":14321.8896484375,\"1754\":16971.015625,\"1755\":16971.015625,\"1756\":16971.015625,\"1757\":16971.015625,\"1758\":16971.015625,\"1759\":16971.015625,\"1760\":16971.015625,\"1761\":16971.015625,\"1762\":16971.015625,\"1763\":14321.8896484375,\"1764\":14321.8896484375,\"1765\":14321.8896484375,\"1766\":16971.015625,\"1767\":16971.015625,\"1768\":14321.8896484375,\"1769\":16971.015625,\"1770\":14321.8896484375,\"1771\":14321.8896484375,\"1772\":14321.8896484375,\"1773\":14321.8896484375,\"1774\":14321.8896484375,\"1775\":16971.015625,\"1776\":16971.015625,\"1777\":16971.015625,\"1778\":14321.8896484375,\"1779\":16971.015625,\"1780\":16971.015625,\"1781\":16971.015625,\"1782\":16971.015625,\"1783\":14321.8896484375,\"1784\":16971.015625,\"1785\":14321.8896484375,\"1786\":16971.015625,\"1787\":16971.015625,\"1788\":14321.8896484375,\"1789\":16971.015625,\"1790\":16971.015625,\"1791\":14321.8896484375,\"1792\":14321.8896484375,\"1793\":16971.015625,\"1794\":16971.015625,\"1795\":16971.015625,\"1796\":14321.8896484375,\"1797\":16971.015625,\"1798\":14321.8896484375,\"1799\":16971.015625,\"1800\":14321.8896484375,\"1801\":16971.015625,\"1802\":16971.015625,\"1803\":16971.015625,\"1804\":14321.8896484375,\"1805\":14321.8896484375,\"1806\":16971.015625,\"1807\":14321.8896484375,\"1808\":14321.8896484375,\"1809\":14321.8896484375,\"1810\":14321.8896484375,\"1811\":14321.8896484375,\"1812\":16971.015625,\"1813\":14321.8896484375,\"1814\":16971.015625,\"1815\":16971.015625,\"1816\":14321.8896484375,\"1817\":14321.8896484375,\"1818\":16971.015625,\"1819\":14321.8896484375,\"1820\":16971.015625,\"1821\":14321.8896484375,\"1822\":14321.8896484375,\"1823\":14321.8896484375,\"1824\":14321.8896484375,\"1825\":16971.015625,\"1826\":16971.015625,\"1827\":16971.015625,\"1828\":16971.015625,\"1829\":16971.015625,\"1830\":16971.015625,\"1831\":14321.8896484375,\"1832\":14321.8896484375,\"1833\":16971.015625,\"1834\":16971.015625,\"1835\":16971.015625,\"1836\":14321.8896484375,\"1837\":14321.8896484375,\"1838\":16971.015625,\"1839\":14321.8896484375,\"1840\":16971.015625,\"1841\":16971.015625,\"1842\":16971.015625,\"1843\":14321.8896484375,\"1844\":14321.8896484375,\"1845\":16971.015625,\"1846\":14321.8896484375,\"1847\":16971.015625,\"1848\":16971.015625,\"1849\":14321.8896484375,\"1850\":14321.8896484375,\"1851\":16971.015625,\"1852\":14321.8896484375,\"1853\":14321.8896484375,\"1854\":14321.8896484375,\"1855\":14321.8896484375,\"1856\":16971.015625,\"1857\":14321.8896484375,\"1858\":14321.8896484375,\"1859\":16971.015625,\"1860\":16971.015625,\"1861\":16971.015625,\"1862\":14321.8896484375,\"1863\":14321.8896484375,\"1864\":16971.015625,\"1865\":14321.8896484375,\"1866\":16971.015625,\"1867\":16971.015625,\"1868\":14321.8896484375,\"1869\":14321.8896484375,\"1870\":16971.015625,\"1871\":14321.8896484375,\"1872\":16971.015625,\"1873\":14321.8896484375,\"1874\":16971.015625,\"1875\":14321.8896484375,\"1876\":16971.015625,\"1877\":14321.8896484375,\"1878\":16971.015625,\"1879\":16971.015625,\"1880\":14321.8896484375,\"1881\":16971.015625,\"1882\":16971.015625,\"1883\":16971.015625,\"1884\":16971.015625,\"1885\":16971.015625,\"1886\":14321.8896484375,\"1887\":16971.015625,\"1888\":16971.015625,\"1889\":14321.8896484375,\"1890\":14321.8896484375,\"1891\":16971.015625,\"1892\":16971.015625,\"1893\":14321.8896484375,\"1894\":14321.8896484375,\"1895\":14321.8896484375,\"1896\":16971.015625,\"1897\":16971.015625,\"1898\":14321.8896484375,\"1899\":14321.8896484375,\"1900\":14321.8896484375,\"1901\":16971.015625,\"1902\":14321.8896484375,\"1903\":14321.8896484375,\"1904\":16971.015625,\"1905\":16971.015625,\"1906\":16971.015625,\"1907\":16971.015625,\"1908\":14321.8896484375,\"1909\":16971.015625,\"1910\":16971.015625,\"1911\":16971.015625,\"1912\":16971.015625,\"1913\":14321.8896484375,\"1914\":14321.8896484375,\"1915\":14321.8896484375,\"1916\":16971.015625,\"1917\":14321.8896484375,\"1918\":14321.8896484375,\"1919\":16971.015625,\"1920\":14321.8896484375,\"1921\":14321.8896484375,\"1922\":16971.015625,\"1923\":16971.015625,\"1924\":14321.8896484375,\"1925\":16971.015625,\"1926\":16971.015625,\"1927\":16971.015625,\"1928\":14321.8896484375,\"1929\":16971.015625,\"1930\":14321.8896484375,\"1931\":16971.015625,\"1932\":16971.015625,\"1933\":14321.8896484375,\"1934\":14321.8896484375,\"1935\":14321.8896484375,\"1936\":14321.8896484375,\"1937\":14321.8896484375,\"1938\":16971.015625,\"1939\":16971.015625,\"1940\":14321.8896484375,\"1941\":16971.015625,\"1942\":14321.8896484375,\"1943\":16971.015625,\"1944\":16971.015625,\"1945\":16971.015625,\"1946\":16971.015625,\"1947\":16971.015625,\"1948\":14321.8896484375,\"1949\":16971.015625,\"1950\":16971.015625,\"1951\":16971.015625,\"1952\":16971.015625,\"1953\":16971.015625,\"1954\":16971.015625,\"1955\":16971.015625,\"1956\":16971.015625,\"1957\":14321.8896484375,\"1958\":14321.8896484375,\"1959\":14321.8896484375,\"1960\":14321.8896484375,\"1961\":14321.8896484375,\"1962\":14321.8896484375,\"1963\":16971.015625,\"1964\":16971.015625,\"1965\":14321.8896484375,\"1966\":14321.8896484375,\"1967\":16971.015625,\"1968\":16971.015625,\"1969\":16971.015625,\"1970\":16971.015625,\"1971\":16971.015625,\"1972\":14321.8896484375,\"1973\":14321.8896484375,\"1974\":16971.015625,\"1975\":16971.015625,\"1976\":16971.015625,\"1977\":16971.015625,\"1978\":16971.015625,\"1979\":16971.015625,\"1980\":16971.015625,\"1981\":14321.8896484375,\"1982\":16971.015625,\"1983\":14321.8896484375,\"1984\":16971.015625,\"1985\":14321.8896484375,\"1986\":14321.8896484375,\"1987\":16971.015625,\"1988\":16971.015625,\"1989\":16971.015625,\"1990\":16971.015625,\"1991\":16971.015625,\"1992\":14321.8896484375,\"1993\":16971.015625,\"1994\":16971.015625,\"1995\":14321.8896484375,\"1996\":14321.8896484375,\"1997\":16971.015625,\"1998\":16971.015625,\"1999\":16971.015625,\"2000\":14321.8896484375,\"2001\":16971.015625,\"2002\":16971.015625,\"2003\":14321.8896484375,\"2004\":16971.015625,\"2005\":14321.8896484375,\"2006\":14321.8896484375,\"2007\":16971.015625,\"2008\":16971.015625,\"2009\":14321.8896484375,\"2010\":14321.8896484375,\"2011\":14321.8896484375,\"2012\":16971.015625,\"2013\":14321.8896484375,\"2014\":16971.015625,\"2015\":14321.8896484375,\"2016\":14321.8896484375,\"2017\":14321.8896484375,\"2018\":16971.015625,\"2019\":16971.015625,\"2020\":14321.8896484375,\"2021\":16971.015625,\"2022\":16971.015625,\"2023\":14321.8896484375,\"2024\":16971.015625,\"2025\":14321.8896484375,\"2026\":14321.8896484375,\"2027\":14321.8896484375,\"2028\":14321.8896484375,\"2029\":14321.8896484375,\"2030\":16971.015625,\"2031\":14321.8896484375,\"2032\":16971.015625,\"2033\":16971.015625,\"2034\":16971.015625,\"2035\":16971.015625,\"2036\":16971.015625,\"2037\":14321.8896484375,\"2038\":16971.015625,\"2039\":16971.015625,\"2040\":14321.8896484375,\"2041\":14321.8896484375,\"2042\":16971.015625,\"2043\":14321.8896484375,\"2044\":16971.015625,\"2045\":16971.015625,\"2046\":14321.8896484375,\"2047\":16971.015625,\"2048\":16971.015625,\"2049\":16971.015625,\"2050\":16971.015625,\"2051\":16971.015625,\"2052\":16971.015625,\"2053\":16971.015625,\"2054\":14321.8896484375,\"2055\":14321.8896484375,\"2056\":14321.8896484375,\"2057\":14321.8896484375,\"2058\":16971.015625,\"2059\":16971.015625,\"2060\":16971.015625,\"2061\":14321.8896484375,\"2062\":14321.8896484375,\"2063\":16971.015625,\"2064\":16971.015625,\"2065\":16971.015625,\"2066\":16971.015625,\"2067\":14321.8896484375,\"2068\":14321.8896484375,\"2069\":14321.8896484375,\"2070\":14321.8896484375,\"2071\":16971.015625,\"2072\":16971.015625,\"2073\":16971.015625,\"2074\":16971.015625,\"2075\":16971.015625,\"2076\":16971.015625,\"2077\":14321.8896484375,\"2078\":16971.015625,\"2079\":16971.015625,\"2080\":16971.015625,\"2081\":16971.015625,\"2082\":16971.015625,\"2083\":14321.8896484375,\"2084\":14321.8896484375,\"2085\":16971.015625,\"2086\":16971.015625,\"2087\":14321.8896484375,\"2088\":16971.015625,\"2089\":16971.015625,\"2090\":14321.8896484375,\"2091\":14321.8896484375,\"2092\":14321.8896484375,\"2093\":16971.015625,\"2094\":16971.015625,\"2095\":14321.8896484375,\"2096\":14321.8896484375,\"2097\":16971.015625,\"2098\":16971.015625,\"2099\":16971.015625,\"2100\":16971.015625,\"2101\":16971.015625,\"2102\":14321.8896484375,\"2103\":16971.015625,\"2104\":14321.8896484375,\"2105\":14321.8896484375,\"2106\":14321.8896484375,\"2107\":16971.015625,\"2108\":16971.015625,\"2109\":16971.015625,\"2110\":16971.015625,\"2111\":14321.8896484375,\"2112\":14321.8896484375,\"2113\":16971.015625,\"2114\":16971.015625,\"2115\":16971.015625,\"2116\":14321.8896484375,\"2117\":16971.015625,\"2118\":14321.8896484375,\"2119\":14321.8896484375,\"2120\":16971.015625,\"2121\":16971.015625,\"2122\":16971.015625,\"2123\":16971.015625,\"2124\":16971.015625,\"2125\":14321.8896484375,\"2126\":16971.015625,\"2127\":16971.015625,\"2128\":14321.8896484375,\"2129\":16971.015625,\"2130\":16971.015625,\"2131\":14321.8896484375,\"2132\":14321.8896484375,\"2133\":16971.015625,\"2134\":14321.8896484375,\"2135\":14321.8896484375,\"2136\":16971.015625,\"2137\":16971.015625,\"2138\":16971.015625,\"2139\":14321.8896484375,\"2140\":16971.015625,\"2141\":14321.8896484375,\"2142\":16971.015625,\"2143\":16971.015625,\"2144\":16971.015625,\"2145\":14321.8896484375,\"2146\":14321.8896484375,\"2147\":16971.015625,\"2148\":14321.8896484375,\"2149\":16971.015625,\"2150\":16971.015625,\"2151\":16971.015625,\"2152\":16971.015625,\"2153\":14321.8896484375,\"2154\":16971.015625,\"2155\":16971.015625,\"2156\":16971.015625,\"2157\":16971.015625,\"2158\":16971.015625,\"2159\":16971.015625,\"2160\":16971.015625,\"2161\":14321.8896484375,\"2162\":16971.015625,\"2163\":16971.015625,\"2164\":14321.8896484375,\"2165\":14321.8896484375,\"2166\":16971.015625,\"2167\":16971.015625,\"2168\":16971.015625,\"2169\":16971.015625,\"2170\":14321.8896484375,\"2171\":16971.015625,\"2172\":16971.015625,\"2173\":16971.015625,\"2174\":16971.015625,\"2175\":14321.8896484375,\"2176\":14321.8896484375,\"2177\":14321.8896484375,\"2178\":16971.015625,\"2179\":14321.8896484375,\"2180\":16971.015625,\"2181\":14321.8896484375,\"2182\":14321.8896484375,\"2183\":14321.8896484375,\"2184\":16971.015625,\"2185\":16971.015625,\"2186\":16971.015625,\"2187\":14321.8896484375,\"2188\":14321.8896484375,\"2189\":16971.015625,\"2190\":16971.015625,\"2191\":16971.015625,\"2192\":14321.8896484375,\"2193\":14321.8896484375,\"2194\":14321.8896484375,\"2195\":14321.8896484375,\"2196\":16971.015625,\"2197\":14321.8896484375,\"2198\":14321.8896484375,\"2199\":16971.015625,\"2200\":16971.015625,\"2201\":14321.8896484375,\"2202\":14321.8896484375,\"2203\":16971.015625,\"2204\":16971.015625,\"2205\":16971.015625,\"2206\":16971.015625,\"2207\":14321.8896484375,\"2208\":14321.8896484375,\"2209\":16971.015625,\"2210\":16971.015625,\"2211\":14321.8896484375,\"2212\":16971.015625,\"2213\":16971.015625,\"2214\":14321.8896484375,\"2215\":14321.8896484375,\"2216\":14321.8896484375,\"2217\":14321.8896484375,\"2218\":16971.015625,\"2219\":16971.015625,\"2220\":14321.8896484375,\"2221\":14321.8896484375,\"2222\":16971.015625,\"2223\":14321.8896484375,\"2224\":14321.8896484375,\"2225\":16971.015625,\"2226\":16971.015625,\"2227\":16971.015625,\"2228\":14321.8896484375,\"2229\":16971.015625,\"2230\":14321.8896484375,\"2231\":16971.015625,\"2232\":14321.8896484375,\"2233\":16971.015625,\"2234\":16971.015625,\"2235\":16971.015625,\"2236\":16971.015625,\"2237\":16971.015625,\"2238\":14321.8896484375,\"2239\":14321.8896484375,\"2240\":16971.015625,\"2241\":16971.015625,\"2242\":16971.015625,\"2243\":14321.8896484375,\"2244\":16971.015625,\"2245\":16971.015625,\"2246\":16971.015625,\"2247\":14321.8896484375,\"2248\":16971.015625,\"2249\":16971.015625,\"2250\":16971.015625,\"2251\":14321.8896484375,\"2252\":16971.015625,\"2253\":14321.8896484375,\"2254\":14321.8896484375,\"2255\":16971.015625,\"2256\":16971.015625,\"2257\":16971.015625,\"2258\":16971.015625,\"2259\":14321.8896484375,\"2260\":16971.015625,\"2261\":16971.015625,\"2262\":14321.8896484375,\"2263\":14321.8896484375,\"2264\":14321.8896484375,\"2265\":14321.8896484375,\"2266\":14321.8896484375,\"2267\":16971.015625,\"2268\":16971.015625,\"2269\":14321.8896484375,\"2270\":14321.8896484375,\"2271\":14321.8896484375,\"2272\":16971.015625,\"2273\":16971.015625,\"2274\":16971.015625,\"2275\":14321.8896484375,\"2276\":14321.8896484375,\"2277\":16971.015625,\"2278\":16971.015625,\"2279\":16971.015625,\"2280\":14321.8896484375,\"2281\":16971.015625,\"2282\":16971.015625,\"2283\":14321.8896484375,\"2284\":16971.015625,\"2285\":16971.015625,\"2286\":16971.015625,\"2287\":14321.8896484375,\"2288\":16971.015625,\"2289\":14321.8896484375,\"2290\":16971.015625,\"2291\":16971.015625,\"2292\":14321.8896484375,\"2293\":14321.8896484375,\"2294\":16971.015625,\"2295\":16971.015625,\"2296\":14321.8896484375,\"2297\":16971.015625,\"2298\":16971.015625,\"2299\":14321.8896484375,\"2300\":16971.015625,\"2301\":14321.8896484375,\"2302\":14321.8896484375}}'"
            ]
          },
          "execution_count": 150,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Creating the function which can take inputs and return prediction\n",
        "def FunctionGeneratePrediction(inp_age , inp_children):\n",
        "\n",
        "    # Creating a data frame for the model input\n",
        "    SampleInputData=pd.DataFrame(\n",
        "     data=[[inp_age , inp_children]],\n",
        "     columns=['age' , 'children'])\n",
        "\n",
        "    # Calling the function defined above using the input parameters\n",
        "    Predictions=FunctionPredictResult(InputData= SampleInputData)\n",
        "\n",
        "    # Returning the predictions\n",
        "    return(Predictions.to_json())\n",
        "\n",
        "# Function call\n",
        "FunctionGeneratePrediction( inp_age=21,\n",
        "                           inp_children=0,\n",
        "                             )"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6rhLPutYVyow"
      },
      "source": [
        "# Web Deployment using Flask Library/Package\n",
        "# Installing the flask library required to create the API\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0ZJja8OKVwE8",
        "outputId": "ea4d5528-0e6f-480f-d6c7-573ece29f85a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: flask in /usr/local/lib/python3.10/dist-packages (2.2.5)\n",
            "Requirement already satisfied: Werkzeug>=2.2.2 in /usr/local/lib/python3.10/dist-packages (from flask) (3.0.2)\n",
            "Requirement already satisfied: Jinja2>=3.0 in /usr/local/lib/python3.10/dist-packages (from flask) (3.1.3)\n",
            "Requirement already satisfied: itsdangerous>=2.0 in /usr/local/lib/python3.10/dist-packages (from flask) (2.2.0)\n",
            "Requirement already satisfied: click>=8.0 in /usr/local/lib/python3.10/dist-packages (from flask) (8.1.7)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from Jinja2>=3.0->flask) (2.1.5)\n"
          ]
        }
      ],
      "source": [
        "!pip install flask"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y5fqN4pSV6NG"
      },
      "source": [
        "# Creating Flask API"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "X-6YUsryV9VH"
      },
      "outputs": [],
      "source": [
        "from flask import Flask, request, jsonify\n",
        "import pickle\n",
        "import pandas as pd\n",
        "import numpy"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "_GNxtVm5WDDc"
      },
      "outputs": [],
      "source": [
        "app = Flask(__name__)\n",
        "\n",
        "@app.route('/prediction_api', methods=[\"GET\"])\n",
        "def prediction_api():\n",
        "    try:\n",
        "        # Getting the paramters from API call\n",
        "        age_value = float(request.args.get('age'))\n",
        "        children_value=float(request.args.get('children'))\n",
        "\n",
        "\n",
        "        # Calling the funtion to get predictions\n",
        "        prediction_from_api=FunctionGeneratePrediction(\n",
        "                                                       inp_age=age_value,\n",
        "                                                       inp_children=children_value\n",
        "                                                )\n",
        "\n",
        "        return (prediction_from_api)\n",
        "\n",
        "    except Exception as e:\n",
        "        return('Something is not right!:'+str(e))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IqbyqEQhWg5J"
      },
      "source": [
        "# Starting the API engine"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VbOG-xjXWh3J",
        "outputId": "ddcb2bf9-fb61-4639-85b0-639287fe71cf"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " * Serving Flask app '__main__'\n",
            " * Debug mode: on\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
            " * Running on http://127.0.0.1:9000\n",
            "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "if __name__ ==\"__main__\":\n",
        "\n",
        "    # Hosting the API in localhost\n",
        "    app.run(host='127.0.0.1', port=9000, threaded=True, debug=True, use_reloader=False)\n",
        "    # Interrupt kernel to stop the API"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uFmujWY5WxUs"
      },
      "source": [
        "'''\n",
        "Sample URL to call the API\n",
        "Copy and paste below URL in the web browser\n",
        "http://127.0.0.1:9000/prediction_api?LSTAT=4.9&RM=6.5&PTRATIO=15.3\n",
        "'''"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SfLHl_Z8W8ki"
      },
      "source": [
        "# Desktop App deployment: Tkinter package\n",
        "* Will not work on Google Colab.\n",
        "* Need to use PyCharm to run this code.\n",
        "* We need to make sure  we include the data file (Medical_insurance.csv)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "mtG199K2XJ1I",
        "outputId": "87042fe5-0119-4f4f-cff5-44bc2a931b57",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 349
        }
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "TclError",
          "evalue": "no display name and no $DISPLAY environment variable",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mTclError\u001b[0m                                  Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-1eff97dc580d>\u001b[0m in \u001b[0;36m<cell line: 44>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     43\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     44\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'__main__'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 45\u001b[0;31m     \u001b[0mroot\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtk\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTk\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     46\u001b[0m     \u001b[0mapp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mHousePricePredictionApp\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mroot\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     47\u001b[0m     \u001b[0mroot\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmainloop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/lib/python3.10/tkinter/__init__.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, screenName, baseName, className, useTk, sync, use)\u001b[0m\n\u001b[1;32m   2297\u001b[0m                 \u001b[0mbaseName\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbaseName\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mext\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2298\u001b[0m         \u001b[0minteractive\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2299\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtk\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_tkinter\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcreate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mscreenName\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbaseName\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mclassName\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minteractive\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mwantobjects\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0museTk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msync\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0muse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2300\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0museTk\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2301\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_loadtk\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mTclError\u001b[0m: no display name and no $DISPLAY environment variable"
          ]
        }
      ],
      "source": [
        "import tkinter as tk\n",
        "from tkinter import messagebox\n",
        "from tkinter import ttk\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from xgboost import XGBRegressor\n",
        "\n",
        "class HousePricePredictionApp:\n",
        "    def __init__(self, master):\n",
        "        self.master = master\n",
        "        self.master.title('Medical Insurance Price Predictiomn')\n",
        "        self.data = pd.read_csv('Medical_insurance.csv')\n",
        "        self.sliders = []\n",
        "\n",
        "        self.X = self.data.drop('charges', axis=1).values\n",
        "        self.y = self.data['charges'].values\n",
        "\n",
        "        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)\n",
        "\n",
        "        self.model = XGBRegressor()\n",
        "        self.model.fit(self.X_train, self.y_train)\n",
        "\n",
        "        self.create_widgets()\n",
        "\n",
        "    def create_widgets(self):\n",
        "        for i, column in enumerate(self.data.columns[:-1]):\n",
        "            label = tk.Label(self.master, text=column + ': ')\n",
        "            label.grid(row=i, column=0)\n",
        "            current_val_label = tk.Label(self.master, text='0.0')\n",
        "            current_val_label.grid(row=i, column=2)\n",
        "            slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient=\"horizontal\",\n",
        "                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))\n",
        "            slider.grid(row=i, column=1)\n",
        "            self.sliders.append((slider, current_val_label))\n",
        "\n",
        "        predict_button = tk.Button(self.master, text='Predict Insurance Charge', command=self.predict_price)\n",
        "        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=3)\n",
        "\n",
        "    def predict_price(self):\n",
        "        inputs = [float(slider.get()) for slider, _ in self.sliders]\n",
        "        price = self.model.predict([inputs])\n",
        "        messagebox.showinfo('Predicted Charges', f'The predicted insurance charges is ${price[0]:.2f}')\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    root = tk.Tk()\n",
        "    app = HousePricePredictionApp(root)\n",
        "    root.mainloop()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vMBeWXleYifR"
      },
      "source": [
        "#  **END OF PROGRAMMING PROJECT**"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
