# Neural Collaborative Filtering Movie Recommender

A Flask web application that provides personalized movie recommendations using a Neural Collaborative Filtering (NCF) model trained on the MovieLens dataset.

## Features

- **Neural Collaborative Filtering**: Uses a hybrid model combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
- **Demographic-based Cold Start**: Initial recommendations based on user demographics
- **Interactive Rating Interface**: Rate movies and get instant personalized recommendations
- **Session-based Tracking**: Maintains user preferences during the session

## Dataset

Uses the MovieLens 100K dataset containing:
- 100,000 ratings
- 943 users
- 1,682 movies

## Deployment on DigitalOcean App Platform

### Prerequisites

1. Push your code to GitHub
2. Make sure `best_model.pth` is committed to the repository
3. Have a DigitalOcean account

### Deployment Steps

#### Option 1: Using the DigitalOcean Dashboard

1. Log in to [DigitalOcean](https://cloud.digitalocean.com/)
2. Navigate to **Apps** in the left sidebar
3. Click **Create App**
4. Select **GitHub** as the source
5. Authorize DigitalOcean to access your repository
6. Select your repository: `JonathanDonald03/DatascienceWebApp`
7. Select branch: `main`
8. DigitalOcean will auto-detect the app using `.digitalocean/app.yaml`
9. Configure environment variables:
   - Add `APP_SECRET_KEY` as a secret (generate with: `python -c "import secrets; print(secrets.token_hex(32))"`)
10. Review and click **Create Resources**

#### Option 2: Using doctl CLI

```bash
# Install doctl
brew install doctl  # macOS

# Authenticate
doctl auth init

# Create app from spec
doctl apps create --spec .digitalocean/app.yaml

# Set secret key
doctl apps update YOUR_APP_ID --env-vars APP_SECRET_KEY=your-generated-secret-key
```

### Environment Variables

Set the following environment variable in DigitalOcean:

- `APP_SECRET_KEY`: Secret key for Flask sessions (generate a secure random string)

To generate a secure key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### App Configuration

The app is configured with:
- **Instance**: Basic (512 MB RAM, 1 vCPU)
- **Workers**: 2 Gunicorn workers with 4 threads each
- **Port**: 8080
- **Auto-deploy**: Enabled on push to main branch

### Accessing Your App

After deployment, DigitalOcean will provide a URL like:
```
https://movie-recommender-xxxxx.ondigitalocean.app
```

## Local Development

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://localhost:8080` in your browser.

### Project Structure

```
.
├── app.py                      # Main Flask application
├── best_model.pth             # Trained Neural CF model weights
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version specification
├── Procfile                  # Process configuration for deployment
├── .digitalocean/
│   └── app.yaml             # DigitalOcean App Platform configuration
├── data/                    # MovieLens dataset files
├── templates/              # HTML templates
│   ├── index.html         # Main recommendation interface
│   └── demographics.html  # User demographics form
└── README.md              # This file
```

## Model Details

The Neural CF model combines:
- **GMF (Generalized Matrix Factorization)**: Captures linear interactions
- **MLP (Multi-Layer Perceptron)**: Captures non-linear interactions
- **Embeddings**: 128-dimensional user and item embeddings
- **Architecture**: [256, 128, 64, 32] MLP layers with 30% dropout

## Technologies

- **Backend**: Flask, Gunicorn
- **ML Framework**: PyTorch (CPU version)
- **Data**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript

## Monitoring

Monitor your app in the DigitalOcean dashboard:
- View logs in real-time
- Check resource usage
- Monitor deployment history
- Set up alerts

## Scaling

To handle more traffic:
1. Go to your app in DigitalOcean
2. Click on the **web** component
3. Adjust the **Instance Size** or **Instance Count**
4. Click **Save** to redeploy

## Troubleshooting

### Build Failures

- Check that all files are committed to Git
- Verify `requirements.txt` is valid
- Check build logs in DigitalOcean dashboard

### Runtime Errors

- Check runtime logs in DigitalOcean
- Verify `best_model.pth` file exists and is accessible
- Ensure SECRET_KEY environment variable is set

### Performance Issues

- Consider upgrading to a larger instance size
- Increase worker count for more concurrent requests
- Enable Redis for session storage (requires additional setup)

## License

This project uses the MovieLens dataset, which is provided by GroupLens Research at the University of Minnesota.

## Support

For issues specific to:
- **DigitalOcean deployment**: Check [DigitalOcean App Platform docs](https://docs.digitalocean.com/products/app-platform/)
- **Application bugs**: Open an issue on GitHub
