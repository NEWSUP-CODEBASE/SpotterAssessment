#!/usr/bin/env python

import os
import sys
import pandas as pd
import time
import pickle
import logging
import asyncio
import hashlib
import argparse
from opencage.geocoder import OpenCageGeocode
import django
import aiohttp
from tqdm import tqdm
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getenv('SYS_PATH'))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SpotterFuelProject.settings")
django.setup()

from django.conf import settings
from django.core.cache import cache

OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')
OPENCAGE_API_KEYS = [OPENCAGE_API_KEY]
CSV_FILE_NAME = os.getenv('CSV_FILE_NAME', "fuel-prices-for-be-assessment_updated.csv")
CSV_FILE_PATH = os.path.join(settings.BASE_DIR, 'FuelRouter', CSV_FILE_NAME)
CACHE_DIR = os.path.join(settings.BASE_DIR, 'FuelRouter', 'cache')
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, 'cache_metadata.pickle')
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 5))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FuelRouter.cache_script")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

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

class ApiKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys[:]
        self.lock = asyncio.Lock()
        self.index = 0
    
    async def get_key(self):
        async with self.lock:
            if not self.api_keys:
                raise Exception("No API keys available")
            key = self.api_keys[self.index]
            self.index = (self.index + 1) % len(self.api_keys)
            return key

@asynccontextmanager
async def get_session():
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT))
    try:
        yield session
    finally:
        await session.close()

async def geocode_address(api_manager, address, retries=0):
    if retries >= MAX_RETRIES:
        return None
    
    if not address or len(address.strip()) < 2:
        return None
    
    cache_key = f'geocode_fwd_{address.replace(" ", "_").replace(",", "")}'
    
    cached_coords = cache.get(cache_key)
    if cached_coords:
        logger.info(f"Found coordinates for {address} in Redis cache")
        return cached_coords
    
    disk_cached_coords = load_from_disk_cache(cache_key)
    if disk_cached_coords:
        logger.info(f"Found coordinates for {address} in disk cache")
        cache.set(cache_key, disk_cached_coords, timeout=None)
        return disk_cached_coords
    
    try:
        api_key = await api_manager.get_key()
        geocoder = OpenCageGeocode(api_key)
        results = geocoder.geocode(address, countrycode='us')
        
        if results and len(results) > 0:
            lat = results[0]['geometry']['lat']
            lng = results[0]['geometry']['lng']
            coords = (lat, lng)
            
            cache.set(cache_key, coords, timeout=None)
            save_to_disk_cache(cache_key, coords)
            logger.info(f"Geocoded {address} to: {coords}")
            
            return coords
        else:
            logger.warning(f"No geocoding results for {address}")
            return None
            
    except Exception as e:
        logger.warning(f"Geocoding error: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
        await asyncio.sleep(2 ** retries)
        return await geocode_address(api_manager, address, retries + 1)

async def update_cache():
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"CSV file not found at {CSV_FILE_PATH}")
        return
    
    logger.info(f"Loading CSV from {CSV_FILE_PATH}")
    df = pd.read_csv(CSV_FILE_PATH)
    
    if "combined_address" not in df.columns:
        df["combined_address"] = df.apply(
            lambda row: f'{row["Truckstop Name"]}, {row["Address"]}, {row["City"]}, {row["State"]}',
            axis=1
        )
    
    if "latitude" not in df.columns:
        df["latitude"] = None
    if "longitude" not in df.columns:
        df["longitude"] = None
    
    last_processed_index = 0
    if os.path.exists(CACHE_METADATA_FILE):
        with open(CACHE_METADATA_FILE, 'rb') as f:
            try:
                metadata = pickle.load(f)
                last_processed_index = metadata.get("last_processed_index", 0)
                logger.info(f"Resuming from index {last_processed_index}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}. Starting from beginning.")
    
    api_manager = ApiKeyManager(OPENCAGE_API_KEYS)
    
    async with get_session():
        for index, row in tqdm(df.iloc[last_processed_index:].iterrows(), 
                              total=len(df) - last_processed_index,
                              desc="Geocoding Progress"):
            
            address = row['combined_address']
            
            coords = await geocode_address(api_manager, address)
            if coords:
                lat, lng = coords
                df.at[index, 'latitude'] = lat
                df.at[index, 'longitude'] = lng
            
            await asyncio.sleep(0.5)
            
            if index % 10 == 0:
                with open(CACHE_METADATA_FILE, 'wb') as f:
                    pickle.dump({"last_processed_index": index + 1}, f)
    
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    df.columns = df.columns.str.replace(' ', '_')
    
    df["Retail_Price"] = pd.to_numeric(df["Retail_Price"], errors="coerce")
    
    logger.info(f"Saving dataframe with {len(df)} locations to Redis cache")
    cache.set('fuel_prices_df', df.to_json(), timeout=None)
    
    logger.info(f"Saving dataframe to disk cache")
    save_to_disk_cache('fuel_prices_df', df)
    
    with open(CACHE_METADATA_FILE, 'wb') as f:
        pickle.dump({"last_processed_index": len(df), "last_update": time.time()}, f)
    
    logger.info("Cache update completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FuelRouter Cache Script")
    parser.add_argument("--force", action="store_true", help="Force cache update")
    args = parser.parse_args()
    
    if args.force and os.path.exists(CACHE_METADATA_FILE):
        logger.info("Force flag set, starting from beginning")
        with open(CACHE_METADATA_FILE, 'wb') as f:
            pickle.dump({"last_processed_index": 0}, f)
    
    asyncio.run(update_cache())