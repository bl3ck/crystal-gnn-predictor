# CrystaLytics

A GNN-powered tool for exploring and predicting material properties using Graph Neural Networks and the Materials Project API.

## Features

- **Real-time Materials Data**: Fetches crystal structures from the Materials Project API
- **Graph Neural Networks**: Predicts material properties using GNN models
- **Explainable AI**: Provides insights into model predictions using GNNExplainer
- **Interactive Visualizations**: 3D crystal structure plots and property comparisons
- **Crystallographic Analysis**: Space group analysis and .cif file downloads

## Setup

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd crystal-gnn-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Materials Project API key:
   - Get a free API key from [Materials Project](https://materialsproject.org/api)
   - Create a `.streamlit/secrets.toml` file:
```toml
MP_API_KEY = "your_api_key_here"
```

4. Run the application:
```bash
streamlit run app.py
```

### Deployment

#### Railway
1. Connect your GitHub repository to Railway
2. Set the environment variable: `MP_API_KEY=your_api_key_here`
3. Deploy using the included `railway.json` configuration

#### Render
1. Connect your GitHub repository to Render
2. Set the environment variable: `MP_API_KEY=your_api_key_here`
3. Deploy using the included `render.yaml` configuration

#### Streamlit Cloud
1. Connect your GitHub repository to Streamlit Cloud
2. Add your API key to the Streamlit secrets management
3. Deploy directly from the Streamlit Cloud dashboard

## Usage

1. **Material Selection**: Choose from pre-loaded samples or fetch materials by ID
2. **Property Prediction**: Use the GNN model to predict material properties
3. **Explainable AI**: View which atoms and bonds influence predictions
4. **Crystallographic Analysis**: Analyze space groups and download .cif files

## Technical Details

The application uses:
- **PyTorch Geometric** for graph neural networks
- **Materials Project API** for crystal structure data
- **Pymatgen** for materials analysis
- **Streamlit** for the web interface
- **Plotly** for interactive visualizations

## License

This project is open source and available under the MIT License.
