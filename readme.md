# SpotterAssessment

## Next-Gen Fuel Route Optimization System

---

## 1. Introduction

### 1.1 Overview

The **Fuel Route Optimization System** is a **high-performance** travel assistant that calculates the most fuel-efficient route between two locations in the USA. The system automatically selects **cost-effective fuel stops**, generates an interactive route map, and provides **real-time cost estimates** for the journey.

> **NOTE:** Due to plan limitations and large CSV size, only a chunk of CSV data is cached and stored (100 Locations). The caching script has multi-async logic where multiple keys allow the complete CSV to be cached and loaded efficiently.

### 1.2 Key Features

- **Cache-powered route optimization for faster responses**
- **Automated fuel stop selection**
- **Interactive route visualization**
- **Optimized performance with caching**
- **Enterprise-grade location finding logic**
- **Best visual experience with a detailed documented guide**

---

## 2. Tech Stack

### Backend Technologies

- **Django + Django REST Framework** – API development
- **Celery + Redis** – Background task execution (*To add Celery for cache updating worker*)
- **GraphHopper API** – Route calculation - [FREE Plan]
- **OpenCage Geocoder API** – Geolocation services - [FREE Plan]

### Frontend & Visualization

- **Folium** – Interactive map generation
- **Modern UI/UX principles applied**
- **HTML, CSS, and JS for Basic UI**
- **Swagger and Redoc integration for better API playground support**

### Data & Intelligence

- **Fuel Prices CSV** – Live fuel price data (*Provided in assessment email*)
- **ML-ready architecture** for predictive analytics
- **In-depth and aggressive algorithms using powerful cache mechanisms**
- **Detailed API for integration**

---

## 3. System Workflow

### 3.1 Route Processing

1. **User enters start and finish locations**
2. System validates locations as **within the USA**
3. **GraphHopper API** generates the optimized route
4. **Fuel stops are optimized** for cost efficiency and spacing
5. System calculates **total fuel cost**
6. **Response includes:**
    - **Route map with fuel stops**
    - **List of fuel stops & real-time prices**
    - **Total trip fuel cost**
7. **Strong cache layer will cache new requests and use cached responses when available**

---

## 4. API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/get-route/` | `GET` | Fetch optimized route data |
| `/route/` | `POST` | Returns full route & fuel stops |
| `/route/simple/` | `GET` | Fetch a simplified fuel route |

---

## 5. Backend Architecture

### 5.1 Performance Optimization

- **Redis caching** for reduced API response times
- **Parallel processing** for rapid computation

### 5.2 Fuel Stop Selection Strategy

- **Prioritizes lowest fuel prices along the route**
- **Ensures no refuel gap exceeds 500 miles**
- **Stops selected within a 10-mile range of the route**

### 5.3 Security & Reliability

- **JWT Authentication** for secure API access
- **Built-in error handling** for stability
- **Scalable architecture** for high-load support

---

## 6. Deployment Strategy

- Hosted on **Ubuntu servers** with **Docker containers**
- **NGINX** for optimized request handling
- **Gunicorn** for scalable WSGI execution

---

## 7. Future Enhancements

- **AI-driven fuel price forecasting**
- **Mobile application integration**
- **User profiles & route history tracking**

---

## 8. Visuals & Documentation

- **Company Logo:** *(Insert image here)*
- **Route Map Example:** *(Insert screenshot here)*
- **Fuel Stop Data Sample:** *(Insert data table here)*
- **Database Schema Diagram:** *(Attach ER diagram here)*

---

## 9. Additional Notes

### Cache Initialization

```sh
mkdir -p FuelRouter/management/commands

touch FuelRouter/management/commands/__init__.py

touch FuelRouter/management/commands/initialize_cache.py
```

---

## 10. Local Setup & Start Project Commands

### Clone the Repository
```sh
git clone <repo-url>
cd SpotterAssessment
```

### Create and Activate a Virtual Environment
```sh
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Apply Migrations
```sh
python manage.py migrate
```

### Run Development Server
```sh
python manage.py runserver
```

---

## 11. Cache Initialization & Execution

### Run the Cache Initialization Script
```sh
python manage.py initialize_cache
```

### Start Celery Worker
```sh
celery -A FuelRouter worker --loglevel=info
```

### Start Redis Server
```sh
redis-server
```

---

## 12. Local CURL Commands

```sh
curl --location 'http://127.0.0.1:8000/api/route/' \
--header 'Content-Type: application/json' \
--data '{
"start_location": "Chicago, IL, USA",
"end_location": "New York, NY, USA"
}'
```

---

## 13. Local UI Demo

Access the UI via:

```sh
http://127.0.0.1:8000/api/spotter-fuel-dashboard/
```

### UI Screenshots:
<a href="https://ibb.co/84jN5fzS"><img src="https://i.ibb.co/Z1hG8FHk/Screenshot-2025-03-01-at-4-08-54-PM.png" alt="Screenshot-2025-03-01-at-4-08-54-PM" border="0"></a>
<a href="https://ibb.co/HWzrWtr"><img src="https://i.ibb.co/g5TD5SD/Screenshot-2025-03-01-at-4-09-11-PM.png" alt="Screenshot-2025-03-01-at-4-09-11-PM" border="0"></a>
<a href="https://ibb.co/MySV455d"><img src="https://i.ibb.co/8ns5p44k/Screenshot-2025-03-01-at-4-10-16-PM.png" alt="Screenshot-2025-03-01-at-4-10-16-PM" border="0"></a>


---

## 14. Notes on Caching

- The UI and API use a **caching layer**, ensuring responses are served in **less than 1s**.
- The **complete CSV** can be cached asynchronously using **multiple keys**.

---

## 15. API Testing

A **Postman collection** is included in the codebase for easier API testing.

---

## 16. Contact & Support

For any questions, please contact **[rcviit4196@gmail.com]**.

Looking forward to working with you!
