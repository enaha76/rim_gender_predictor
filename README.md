# Gender Predictor API

A FastAPI application for predicting gender from names using machine learning.

## Local Development

Run the application locally using uvicorn:

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Deployment

### Prerequisites

- Docker and Docker Compose installed
- Machine learning model files (`model.pkl` and `label_encoder.pkl`) in the correct location:
  - `models/gender_prediction_model/gender_prediction_model/model.pkl`
  - `models/gender_prediction_model/gender_prediction_model/label_encoder.pkl`

### Using Docker Compose (Recommended)

1. Make sure your `.env` file is configured with the appropriate database settings:

```
DB_USER=genderapp
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gender_predictions
```

2. Build and start the containers:

```bash
docker-compose up -d
```

3. Access the API at `http://localhost:8000`
4. View the API documentation at `http://localhost:8000/docs`

### Using Docker (Without Database)

1. Build the Docker image:

```bash
docker build -t gender-predictor .
```

2. Run the container:

```bash
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models gender-predictor
```

3. Access the API at `http://localhost:8000`

### Cloud Deployment

#### Heroku

1. Install Heroku CLI and login:

```bash
heroku login
```

2. Create a new Heroku app:

```bash
heroku create your-gender-predictor-app
```

3. Add a Postgres database:

```bash
heroku addons:create heroku-postgresql:hobby-dev
```

4. Deploy the app:

```bash
git push heroku main
```

#### AWS Elastic Beanstalk

1. Install the EB CLI:

```bash
pip install awsebcli
```

2. Initialize EB application:

```bash
eb init
```

3. Create an environment and deploy:

```bash
eb create gender-predictor-env
```

## API Endpoints

- `GET /`: Root endpoint
- `POST /predict`: Predict gender for a single name
- `POST /predict-batch`: Predict gender for multiple names
- `GET /health`: Health check endpoint
- `GET /model-info`: Get information about the ML model

## Testing

Run the test script to test the API:

```bash
python test_api.py
```