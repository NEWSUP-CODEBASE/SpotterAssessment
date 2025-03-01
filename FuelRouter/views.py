from django.shortcuts import render
import requests
import folium
from opencage.geocoder import OpenCageGeocode
from geopy.distance import geodesic
import pandas as pd
import os
import json
import pickle
from django.conf import settings
from django.core.cache import cache
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.exceptions import ValidationError
from django.http import JsonResponse
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import asyncio
import concurrent.futures
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_cookie
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FuelRouter.views')




GRAPHHOPPER_API_KEY = os.getenv("GRAPHHOPPER_API_KEY")
FUEL_EFFICIENCY = int(os.getenv("FUEL_EFFICIENCY", 10))
MAX_MILES_WITHOUT_REFUEL = int(os.getenv("MAX_MILES_WITHOUT_REFUEL", 500))
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")


CSV_FILE_NAME = "fuel-prices-for-be-assessment_updated.csv"
CSV_FILE_PATH = os.path.join(settings.BASE_DIR, 'FuelRouter', CSV_FILE_NAME)
CACHE_DIR = os.path.join(settings.BASE_DIR, 'FuelRouter', 'cache')

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
executor = ThreadPoolExecutor(max_workers=10)

_fuel_prices_df = None

def save_to_disk_cache(key, data):
    try:
        cache_file_path = os.path.join(CACHE_DIR, f"{key}.pickle")
        with open(cache_file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving to disk cache: {e}")
        return False

def load_from_disk_cache(key):
    try:
        cache_file_path = os.path.join(CACHE_DIR, f"{key}.pickle")
        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading from disk cache: {e}")
    return None

def get_fuel_prices_df():
    global _fuel_prices_df
    
    if _fuel_prices_df is not None:
        return _fuel_prices_df
    
    df_serialized = cache.get('fuel_prices_df')
    if df_serialized is not None:
        try:
            df = pd.read_json(df_serialized)
            logger.info(f"Loaded fuel prices from Redis cache with {len(df)} locations")
            return df
        except Exception as e:
            logger.error(f"Error loading from Redis: {e}")
    
    df_from_disk = load_from_disk_cache('fuel_prices_df')
    if df_from_disk is not None:
        try:
            cache.set('fuel_prices_df', df_from_disk.to_json(), timeout=None)
            logger.info(f"Loaded fuel prices from disk cache with {len(df_from_disk)} locations")
            return df_from_disk
        except Exception as e:
            logger.error(f"Error loading disk cache to Redis: {e}")
    
    logger.info("Fuel prices not in any cache, loading CSV...")
    try:
        if not os.path.exists(CSV_FILE_PATH):
            logger.error(f"CSV file not found at {CSV_FILE_PATH}")
            return None
            
        df = pd.read_csv(CSV_FILE_PATH)
        
        if "combined_address" not in df.columns:
            df["combined_address"] = df.apply(
                lambda row: f'{row["Truckstop Name"]}, {row["Address"]}, {row["City"]}, {row["State"]}',
                axis=1
            )
        
        if "latitude" not in df.columns or "longitude" not in df.columns:
            df["latitude"] = None
            df["longitude"] = None
            
            addresses = df["combined_address"].tolist()
            batch_size = 10
            
            for i in range(0, len(addresses), batch_size):
                batch = addresses[i:i+batch_size]
                results = []
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    results = list(executor.map(forward_geocode, batch))
                
                for j, coords in enumerate(results):
                    if coords:
                        df.loc[i+j, "latitude"] = coords[0]
                        df.loc[i+j, "longitude"] = coords[1]
                
                time.sleep(1)
        
        df.dropna(subset=["latitude", "longitude"], inplace=True)
        df.columns = df.columns.str.replace(' ', '_')
        df["Retail_Price"] = pd.to_numeric(df["Retail_Price"], errors="coerce")
        
        if not df.empty:
            try:
                cache.set('fuel_prices_df', df.to_json(), timeout=None)
                save_to_disk_cache('fuel_prices_df', df)
                logger.info(f"CSV loaded and cached with {len(df)} valid locations")
            except Exception as e:
                logger.error(f"Error caching fuel data: {e}")
        else:
            logger.warning("No valid fuel stop locations found in CSV")
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return None
    
    return df

@lru_cache(maxsize=10000)
def _geocode_cached(address):
    if not address or len(address.strip()) < 2:
        return None
    
    try:
        time.sleep(0.5)
        results = geocoder.geocode(address, countrycode='us')
        
        if results and len(results) > 0:
            lat = results[0]['geometry']['lat']
            lng = results[0]['geometry']['lng']
            return (lat, lng)
        else:
            return None
    except Exception as e:
        logger.error(f"Error geocoding {address}: {e}")
        return None

def forward_geocode(address):
    if not address or len(address.strip()) < 2:
        return None
        
    cache_key = f'geocode_fwd_{address.replace(" ", "_").replace(",", "")}'
    
    cached_coords = cache.get(cache_key)
    if cached_coords:
        return cached_coords
    
    disk_cached_coords = load_from_disk_cache(cache_key)
    if disk_cached_coords:
        cache.set(cache_key, disk_cached_coords, timeout=None)
        return disk_cached_coords
    
    coords = _geocode_cached(address)
    if coords:
        cache.set(cache_key, coords, timeout=None)
        save_to_disk_cache(cache_key, coords)
        logger.info(f"Forward geocoded {address} to: {coords}")
    
    return coords

@lru_cache(maxsize=1000)
def _reverse_geocode_cached(lat, lng):
    try:
        time.sleep(0.5)
        results = geocoder.reverse_geocode(lat, lng)
        
        if results and len(results) > 0:
            return results[0]['formatted']
        else:
            return None
    except Exception as e:
        logger.error(f"Error reverse geocoding ({lat}, {lng}): {e}")
        return None

def reverse_geocode(lat, lng):
    lat_rounded = round(lat, 6)
    lng_rounded = round(lng, 6)
    
    cache_key = f'geocode_rev_{lat_rounded}_{lng_rounded}'
    
    cached_address = cache.get(cache_key)
    if cached_address:
        return cached_address
    
    disk_cached_address = load_from_disk_cache(cache_key)
    if disk_cached_address:
        cache.set(cache_key, disk_cached_address, timeout=None)
        return disk_cached_address
    
    formatted_address = _reverse_geocode_cached(lat_rounded, lng_rounded)
    if formatted_address:
        cache.set(cache_key, formatted_address, timeout=None)
        save_to_disk_cache(cache_key, formatted_address)
        logger.info(f"Reverse geocoded ({lat_rounded}, {lng_rounded}) to: {formatted_address}")
    
    return formatted_address

def get_route(start_lat, start_lon, end_lat, end_lon, api_key):
    start_lat_r = round(start_lat, 4)
    start_lon_r = round(start_lon, 4)
    end_lat_r = round(end_lat, 4)
    end_lon_r = round(end_lon, 4)
    
    cache_key = f"route_{start_lat_r}_{start_lon_r}_{end_lat_r}_{end_lon_r}"
    
    cached_route = cache.get(cache_key)
    if cached_route:
        logger.info("Using route from Redis cache")
        return cached_route
    
    disk_cached_route = load_from_disk_cache(cache_key)
    if disk_cached_route:
        cache.set(cache_key, disk_cached_route, timeout=None)
        logger.info("Using route from disk cache")
        return disk_cached_route
    
    url = os.getenv("GRAPHHOPPER_BASE_URL")
    params = {
        "point": [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
        "profile": "car",
        "locale": "en",
        "calc_points": True,
        "key": api_key,
        "points_encoded": False,
        "instructions": True,
        "type": "json"
    }
    
    try:
        logger.info(f"Requesting route from GraphHopper API: {start_lat},{start_lon} to {end_lat},{end_lon}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"GraphHopper API request failed: {response.status_code}")
            return None
            
        data = response.json()
        
        if "paths" in data and data["paths"]:
            route = data["paths"][0]
            cache.set(cache_key, route, timeout=None)
            save_to_disk_cache(cache_key, route)
            return route
        else:
            return None
    except Exception as e:
        logger.error(f"Error in route API: {e}")
        return None

def create_map(route_geometry, start_lat, start_lon, end_lat, end_lon, fuel_stops):
    if not route_geometry:
        return None
    
    try:
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, prefer_canvas=True)
        
        simplified_coords = [[coord[1], coord[0]] for i, coord in enumerate(route_geometry) if i % 4 == 0]
        
        folium.PolyLine(
            simplified_coords, 
            color="#2ecc71", 
            weight=4, 
            opacity=0.8
        ).add_to(m)
        
        start_address = reverse_geocode(start_lat, start_lon) or "Start Location"
        end_address = reverse_geocode(end_lat, end_lon) or "Destination"
        
        folium.Marker(
            [start_lat, start_lon],
            popup=f"<b>Start:</b><br>{start_address}",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
        
        folium.Marker(
            [end_lat, end_lon],
            popup=f"<b>Destination:</b><br>{end_address}",
            icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')
        ).add_to(m)
        
        for i, stop in enumerate(fuel_stops):
            if "latitude" in stop and "longitude" in stop and stop["latitude"] and stop["longitude"]:
                address = stop.get("combined_address", f"Fuel Stop #{i+1}")
                
                popup_content = f'''
                    <b>{stop.get("Truckstop_Name", f"Fuel Stop #{i+1}")}</b><br>
                    Price: ${stop.get("Retail_Price", "N/A")}<br>
                    {address}
                '''
                
                folium.Marker(
                    [stop["latitude"], stop["longitude"]],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='blue', icon='gas-pump', prefix='fa')
                ).add_to(m)
        
        return m._repr_html_()
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        return None

def calculate_distance(lat1, lon1, lat2, lon2):
    try:
        distance_km = geodesic((lat1, lon1), (lat2, lon2)).km
        return distance_km * 0.621371
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return 0

def calculate_distance_along_route(point, route_points):
    min_distance = float('inf')
    closest_idx = 0
    
    sampled_points = route_points[::10]
    
    for i, (lon, lat) in enumerate(sampled_points):
        dist = calculate_distance(point[0], point[1], lat, lon)
        if dist < min_distance:
            min_distance = dist
            closest_idx = i * 10
    
    start_idx = max(0, closest_idx - 10)
    end_idx = min(len(route_points), closest_idx + 10)
    
    for i in range(start_idx, end_idx):
        if i < len(route_points):
            lon, lat = route_points[i]
            dist = calculate_distance(point[0], point[1], lat, lon)
            if dist < min_distance:
                min_distance = dist
                closest_idx = i
    
    return min_distance, closest_idx

def plan_fuel_stops(route_data, fuel_prices_df, start_lat, start_lon, end_lat, end_lon):
    start_lat_r = round(start_lat, 4)
    start_lon_r = round(start_lon, 4)
    end_lat_r = round(end_lat, 4)
    end_lon_r = round(end_lon, 4)
    
    cache_key = f"fuel_stops_{start_lat_r}_{start_lon_r}_{end_lat_r}_{end_lon_r}"
    
    cached_stops = cache.get(cache_key)
    if cached_stops:
        logger.info("Using fuel stops from Redis cache")
        return cached_stops
    
    disk_cached_stops = load_from_disk_cache(cache_key)
    if disk_cached_stops:
        cache.set(cache_key, disk_cached_stops, timeout=None)
        logger.info("Using fuel stops from disk cache")
        return disk_cached_stops
    
    if fuel_prices_df is None or fuel_prices_df.empty:
        return []
    
    try:
        route_distance_miles = route_data["distance"] / 1000 * 0.621371
        
        if route_distance_miles <= MAX_MILES_WITHOUT_REFUEL:
            logger.info(f"Route distance ({route_distance_miles:.1f} miles) is within fuel range, no stops needed.")
            return []
        
        route_points = route_data["points"]["coordinates"]
        num_stops_needed = int(np.ceil(route_distance_miles / MAX_MILES_WITHOUT_REFUEL))
        segment_length = len(route_points) / (num_stops_needed + 1)
        
        progress_points = []
        for i in range(1, num_stops_needed + 1):
            idx = int(i * segment_length)
            if idx < len(route_points):
                lon, lat = route_points[idx]
                progress_points.append((lat, lon, idx))
        
        fuel_prices_df = fuel_prices_df[['Truckstop_Name', 'Address', 'City', 'State', 'Retail_Price', 'latitude', 'longitude', 'combined_address']].copy()
        fuel_prices_df['route_distance'] = 999999.0
        fuel_prices_df['route_index'] = -1
        
        route_bounding_box = {
            'min_lat': min(start_lat, end_lat) - 0.5,
            'max_lat': max(start_lat, end_lat) + 0.5,
            'min_lon': min(start_lon, end_lon) - 0.5,
            'max_lon': max(start_lon, end_lon) + 0.5
        }
        
        filtered_stops = fuel_prices_df[
            (fuel_prices_df['latitude'] >= route_bounding_box['min_lat']) &
            (fuel_prices_df['latitude'] <= route_bounding_box['max_lat']) &
            (fuel_prices_df['longitude'] >= route_bounding_box['min_lon']) &
            (fuel_prices_df['longitude'] <= route_bounding_box['max_lon'])
        ]
        
        if filtered_stops.empty:
            logger.info("No fuel stops found in route vicinity")
            return []
        
        def process_row(row):
            if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
                min_dist, closest_idx = calculate_distance_along_route(
                    (row['latitude'], row['longitude']), 
                    route_points
                )
                if min_dist <= 10:
                    return row.name, float(min_dist), int(closest_idx)
            return None
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_row, [filtered_stops.iloc[i] for i in range(len(filtered_stops))]))
        
        for result in results:
            if result:
                idx, min_dist, closest_idx = result
                fuel_prices_df.loc[idx, 'route_distance'] = min_dist
                fuel_prices_df.loc[idx, 'route_index'] = closest_idx
        
        nearby_stops = fuel_prices_df[fuel_prices_df['route_index'] >= 0].copy()
        
        if nearby_stops.empty:
            return []
        
        nearby_stops = nearby_stops.sort_values('route_index')
        selected_stops = []
        start_point = (start_lat, start_lon)
        end_point = (end_lat, end_lon)
        current_position = start_point
        remaining_range = MAX_MILES_WITHOUT_REFUEL
        
        for point in progress_points:
            target_lat, target_lon, target_idx = point
            
            candidates = nearby_stops[
                (nearby_stops['route_index'] > 0) & 
                (nearby_stops['route_index'] <= target_idx)
            ]
            
            if candidates.empty:
                continue
                
            distance_to_point = calculate_distance(
                current_position[0], current_position[1], 
                target_lat, target_lon
            )
            
            if distance_to_point >= remaining_range:
                best_stop = candidates.sort_values('Retail_Price').iloc[0]
                selected_stops.append(best_stop.to_dict())
                current_position = (best_stop['latitude'], best_stop['longitude'])
                remaining_range = MAX_MILES_WITHOUT_REFUEL
                nearby_stops = nearby_stops[nearby_stops.index != best_stop.name]
        
        distance_to_end = calculate_distance(
            current_position[0], current_position[1], 
            end_point[0], end_point[1]
        )
        
        if distance_to_end > remaining_range:
            remaining_candidates = nearby_stops[nearby_stops['route_index'] > 0]
            
            if not remaining_candidates.empty:
                best_final_stop = remaining_candidates.sort_values('Retail_Price').iloc[0]
                selected_stops.append(best_final_stop.to_dict())
        
        logger.info(f"Planned {len(selected_stops)} fuel stops for route of {route_distance_miles:.1f} miles")
        
        cache.set(cache_key, selected_stops, timeout=None)
        save_to_disk_cache(cache_key, selected_stops)
        
        return selected_stops
        
    except Exception as e:
        logger.error(f"Error planning fuel stops: {e}")
        return []

def calculate_fuel_cost(route_data, fuel_efficiency, fuel_stops):
    try:
        distance_miles = route_data["distance"] / 1000 * 0.621371
        fuel_needed = distance_miles / fuel_efficiency
        
        if not fuel_stops:
            return fuel_needed * 3.50
        
        total_cost = 0
        total_fuel = fuel_needed
        remaining_fuel = total_fuel
        
        for i, stop in enumerate(fuel_stops):
            try:
                price = float(stop.get("Retail_Price", 3.50))
                
                if i < len(fuel_stops) - 1:
                    fuel_to_fill = min(remaining_fuel, MAX_MILES_WITHOUT_REFUEL/fuel_efficiency)
                else:
                    fuel_to_fill = remaining_fuel
                
                total_cost += fuel_to_fill * price
                remaining_fuel -= fuel_to_fill
                
            except (ValueError, KeyError) as e:
                total_cost += (remaining_fuel / len(fuel_stops)) * 3.50
        
        logger.info(f"Calculated total fuel cost: ${total_cost:.2f} for {distance_miles:.1f} miles")
        return total_cost
    except Exception as e:
        logger.error(f"Error calculating fuel cost: {e}")
        try:
            distance_miles = route_data["distance"] / 1000 * 0.621371
            return (distance_miles / fuel_efficiency) * 3.50
        except:
            return 0

def check_redis_connection():
    try:
        cache.set('redis_test', 'test_value', timeout=10)
        test_value = cache.get('redis_test')
        return test_value == 'test_value'
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        return False

def spotter_fuel_view(request):
    redis_available = check_redis_connection()
    if not redis_available:
        logger.warning("Redis cache is not available, falling back to disk cache")
        
    context = {
        "map_html": None,
        "fuel_stops": [],
        "total_fuel_cost": 0,
        "num_fuel_stops": 0,
        "error_message": None,
        "geocoding_error": None,
        "api_error": None,
        "csv_error": None,
        "route_info": None,
        "start_address": "",
        "end_address": ""
    }
    
    try:
        fuel_prices_df = _fuel_prices_df if _fuel_prices_df is not None else get_fuel_prices_df()
        if fuel_prices_df is None or fuel_prices_df.empty:
            context["csv_error"] = "Fuel price data could not be loaded. Using estimated prices."
    except Exception as e:
        logger.error(f"Error loading fuel data: {e}")
        context["csv_error"] = "Fuel price data could not be loaded. Using estimated prices."
        fuel_prices_df = None
    
    if request.method == "POST":
        start_address = request.POST.get("start_address", "")
        end_address = request.POST.get("end_address", "")
        
        context["start_address"] = start_address
        context["end_address"] = end_address
        
        if not (start_address.lower().endswith("usa") or start_address.lower().endswith("us")):
            start_address += ", USA"
            
        if not (end_address.lower().endswith("usa") or end_address.lower().endswith("us")):
            end_address += ", USA"
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            start_coords_future = executor.submit(forward_geocode, start_address)
            end_coords_future = executor.submit(forward_geocode, end_address)
            
            start_coords = start_coords_future.result()
            end_coords = end_coords_future.result()
        
        if not start_coords:
            context["geocoding_error"] = f"Could not find location: {start_address}"
            return render(request, "index.html", context)
            
        if not end_coords:
            context["geocoding_error"] = f"Could not find location: {end_address}"
            return render(request, "index.html", context)
        
        start_lat, start_lon = start_coords
        end_lat, end_lon = end_coords
        
        try:
            route_data = get_route(start_lat, start_lon, end_lat, end_lon, GRAPHHOPPER_API_KEY)
            if not route_data:
                context["api_error"] = "Could not find a route between these locations. Please check your addresses."
                return render(request, "index.html", context)
        except Exception as e:
            logger.error(f"Error getting route: {e}")
            context["api_error"] = "Error connecting to routing service. Please try again later."
            return render(request, "index.html", context)
        
        route_geometry = route_data["points"]["coordinates"]
        distance_miles = route_data["distance"] / 1000 * 0.621371
        time_minutes = route_data["time"] / (1000 * 60)
        
        context["route_info"] = {
            "distance": round(distance_miles, 1),
            "time": round(time_minutes, 1)
        }
        
        try:
            if fuel_prices_df is None or fuel_prices_df.empty:
                fuel_stops = []
                context["csv_error"] = "Using estimated fuel prices since fuel stop data is unavailable."
            else:
                fuel_stops = plan_fuel_stops(route_data, fuel_prices_df, start_lat, start_lon, end_lat, end_lon)
                
            context["num_fuel_stops"] = len(fuel_stops)
            
            total_fuel_cost = calculate_fuel_cost(route_data, FUEL_EFFICIENCY, fuel_stops)
            context["total_fuel_cost"] = total_fuel_cost
            
            map_html = create_map(route_geometry, start_lat, start_lon, end_lat, end_lon, fuel_stops)
            context["map_html"] = map_html
            
            fuel_stops_data = []
            for i, stop in enumerate(fuel_stops):
                fuel_stops_data.append({
                    "stop_number": i + 1,
                    "Truckstop_Name": stop.get("Truckstop_Name", f"Fuel Stop #{i+1}"),
                    "Retail_Price": stop.get("Retail_Price", "3.50"),
                    "Address": stop.get("Address", "Address unavailable"),
                    "City": stop.get("City", ""),
                    "State": stop.get("State", ""),
                    "latitude": stop.get("latitude", 0),
                    "longitude": stop.get("longitude", 0)
                })
            
            context["fuel_stops"] = fuel_stops_data
            
        except Exception as e:
            logger.error(f"Error planning route: {e}")
            context["error_message"] = "An error occurred while planning your route. Please try again."
            
    return render(request, "index.html", context)

def preload_fuel_prices():
    global _fuel_prices_df
    _fuel_prices_df = get_fuel_prices_df()
    logger.info(f"Preloaded fuel prices with {len(_fuel_prices_df) if _fuel_prices_df is not None else 0} locations")

preload_fuel_prices()


class FuelRouteAPIView(APIView):
    @method_decorator(cache_page(60 * 60 * 24))
    @swagger_auto_schema(
    operation_description="Calculate fuel route between two locations",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['start_location', 'end_location'],
        properties={
            'start_location': openapi.Schema(type=openapi.TYPE_STRING, description='Starting location (within USA)'),
            'end_location': openapi.Schema(type=openapi.TYPE_STRING, description='Destination location (within USA)'),
        },
    ),
    responses={
        200: openapi.Response(
            description="Successful Response",
            schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'route_info': openapi.Schema(type=openapi.TYPE_OBJECT),
                    'fuel_stops': openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_OBJECT)
                    ),
                    'total_fuel_cost': openapi.Schema(type=openapi.TYPE_NUMBER),
                    'map_data': openapi.Schema(type=openapi.TYPE_OBJECT),
                    'status': openapi.Schema(type=openapi.TYPE_STRING),
                }
            )
        ),
        400: "Invalid Input",
        500: "Server Error",
    }
)
    def post(self, request, format=None):
        start_time = time.time()
        logger.info("API request received")
        
        try:
            start_location = request.data.get('start_location')
            end_location = request.data.get('end_location')
            
            if not start_location or not end_location:
                raise ValidationError({"error": "Both start_location and end_location are required"})
                
            if len(start_location.strip()) < 3 or len(end_location.strip()) < 3:
                raise ValidationError({"error": "Locations must be at least 3 characters"})
            
            if not (start_location.lower().endswith("usa") or start_location.lower().endswith("us")):
                start_location += ", USA"
                
            if not (end_location.lower().endswith("usa") or end_location.lower().endswith("us")):
                end_location += ", USA"
            
            fuel_prices_df = _fuel_prices_df if _fuel_prices_df is not None else get_fuel_prices_df()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                start_coords_future = executor.submit(forward_geocode, start_location)
                end_coords_future = executor.submit(forward_geocode, end_location)
                
                start_coords = start_coords_future.result()
                end_coords = end_coords_future.result()
            
            if not start_coords:
                return Response(
                    {"error": f"Could not find location: {start_location}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            if not end_coords:
                return Response(
                    {"error": f"Could not find location: {end_location}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            start_lat, start_lon = start_coords
            end_lat, end_lon = end_coords
            
            route_data = get_route(start_lat, start_lon, end_lat, end_lon, GRAPHHOPPER_API_KEY)
            if not route_data:
                return Response(
                    {"error": "Could not find a route between these locations"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            route_geometry = route_data["points"]["coordinates"]
            distance_miles = route_data["distance"] / 1000 * 0.621371
            time_minutes = route_data["time"] / (1000 * 60)
            
            route_info = {
                "distance_miles": round(distance_miles, 1),
                "time_minutes": round(time_minutes, 1),
                "start_location": start_location,
                "end_location": end_location,
                "start_coordinates": {"lat": start_lat, "lng": start_lon},
                "end_coordinates": {"lat": end_lat, "lng": end_lon}
            }
            
            fuel_stops = []
            if fuel_prices_df is not None and not fuel_prices_df.empty:
                fuel_stops = plan_fuel_stops(route_data, fuel_prices_df, start_lat, start_lon, end_lat, end_lon)
            
            total_fuel_cost = calculate_fuel_cost(route_data, FUEL_EFFICIENCY, fuel_stops)
            
            formatted_fuel_stops = []
            for i, stop in enumerate(fuel_stops):
                formatted_fuel_stops.append({
                    "stop_number": i + 1,
                    "name": stop.get("Truckstop_Name", f"Fuel Stop #{i+1}"),
                    "price": float(stop.get("Retail_Price", 3.50)),
                    "address": stop.get("Address", "Address unavailable"),
                    "city": stop.get("City", ""),
                    "state": stop.get("State", ""),
                    "coordinates": {
                        "lat": float(stop.get("latitude", 0)),
                        "lng": float(stop.get("longitude", 0))
                    }
                })
            
            map_data = {
                "route_geometry": route_geometry,
                "start_coordinates": {"lat": start_lat, "lng": start_lon},
                "end_coordinates": {"lat": end_lat, "lng": end_lon},
                "fuel_stops": [
                    {
                        "coordinates": {"lat": float(stop.get("latitude", 0)), "lng": float(stop.get("longitude", 0))},
                        "name": stop.get("Truckstop_Name", f"Fuel Stop #{i+1}"),
                        "price": float(stop.get("Retail_Price", 3.50))
                    } for i, stop in enumerate(fuel_stops)
                ]
            }
            
            response_data = {
                "route_info": route_info,
                "fuel_stops": formatted_fuel_stops,
                "total_fuel_cost": round(total_fuel_cost, 2),
                "map_data": map_data,
                "fuel_efficiency_mpg": FUEL_EFFICIENCY,
                "status": "success"
            }
            
            processing_time = time.time() - start_time
            logger.info(f"API request processed in {processing_time:.2f} seconds")
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"API error: {str(e)}", exc_info=True)
            return Response(
                {"error": "An unexpected error occurred. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
@cache_page(60 * 60 * 24)
def fuel_route_get(request):
    start_time = time.time()
    logger.info("GET API request received")
    
    try:
        start_location = request.query_params.get('start')
        end_location = request.query_params.get('end')
        
        if not start_location or not end_location:
            return Response(
                {"error": "Both 'start' and 'end' parameters are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        if not (start_location.lower().endswith("usa") or start_location.lower().endswith("us")):
            start_location += ", USA"
            
        if not (end_location.lower().endswith("usa") or end_location.lower().endswith("us")):
            end_location += ", USA"
        
        fuel_prices_df = _fuel_prices_df if _fuel_prices_df is not None else get_fuel_prices_df()
        
        start_coords = forward_geocode(start_location)
        if not start_coords:
            return Response(
                {"error": f"Could not find location: {start_location}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        end_coords = forward_geocode(end_location)
        if not end_coords:
            return Response(
                {"error": f"Could not find location: {end_location}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        start_lat, start_lon = start_coords
        end_lat, end_lon = end_coords
        
        route_data = get_route(start_lat, start_lon, end_lat, end_lon, GRAPHHOPPER_API_KEY)
        if not route_data:
            return Response(
                {"error": "Could not find a route between these locations"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        distance_miles = route_data["distance"] / 1000 * 0.621371
        time_minutes = route_data["time"] / (1000 * 60)
        
        fuel_stops = []
        if fuel_prices_df is not None and not fuel_prices_df.empty:
            fuel_stops = plan_fuel_stops(route_data, fuel_prices_df, start_lat, start_lon, end_lat, end_lon)
        
        total_fuel_cost = calculate_fuel_cost(route_data, FUEL_EFFICIENCY, fuel_stops)
        
        formatted_fuel_stops = []
        for i, stop in enumerate(fuel_stops):
            formatted_fuel_stops.append({
                "stop_number": i + 1,
                "name": stop.get("Truckstop_Name", f"Fuel Stop #{i+1}"),
                "price": float(stop.get("Retail_Price", 3.50)),
                "location": f"{stop.get('City', '')}, {stop.get('State', '')}",
                "lat": float(stop.get("latitude", 0)),
                "lng": float(stop.get("longitude", 0))
            })
        
        response_data = {
            "start": start_location,
            "end": end_location,
            "distance_miles": round(distance_miles, 1),
            "time_minutes": round(time_minutes, 1),
            "fuel_stops": formatted_fuel_stops,
            "total_fuel_cost": round(total_fuel_cost, 2),
            "fuel_efficiency_mpg": FUEL_EFFICIENCY
        }
        
        processing_time = time.time() - start_time
        logger.info(f"GET API request processed in {processing_time:.2f} seconds")
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )