import os
import googlemaps
import pandas as pd

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

if not google_maps_api_key:
    raise ValueError("Google Maps API key must be set in environment variables.")

# Initialize the Google Maps client
gmaps = googlemaps.Client(key=google_maps_api_key)

# List of assurance company names to search for
assurance_companies_names = [
    "ALLIANZ", "AtlantaSanad", "AXA", "CAT", "MAMDA", 
    "MAROCAINE VIE", "MATU", "RMA", "Saham", "Wafa assurance"
]

# Coordinates for Rabat, Casablanca, and Tangier
city_coords = {
    "Rabat": (34.020882, -6.841650),
    "Casablanca": (33.573110, -7.589843),
    "Tangier": (35.759465, -5.833954)
}

# Function to get places
def get_assurance_companies(names, locations, radius=20000):
    place_details = []

    for city, coords in locations.items():
        for name in names:
            search_query = name
            
            # Initialize next_page_token for pagination
            next_page_token = None
            
            # Loop to handle pagination
            while True:
                # Perform the search with optional page token
                if next_page_token:
                    places_result = gmaps.places(query=search_query, location=coords, radius=radius, page_token=next_page_token)
                else:
                    places_result = gmaps.places(query=search_query, location=coords, radius=radius)
                
                # Extract the place IDs
                place_ids = [place['place_id'] for place in places_result['results']]
                
                # Retrieve details for each place
                for place_id in place_ids:
                    details = gmaps.place(place_id=place_id)
                    details['result']['searched_city'] = city
                    details['result']['searched_name'] = name
                    place_details.append(details['result'])
                
                # Check if there are more results to fetch
                next_page_token = places_result.get('next_page_token')
                if not next_page_token:
                    break
    
    # Filter and extract required fields
    filtered_places = []
    for place in place_details:
        filtered_places.append({
            'name': place.get('name'),
            'searched_name': place.get('searched_name'),
            'city': place.get('searched_city'),
            'address': place.get('formatted_address'),
            'latitude': place['geometry']['location'].get('lat'),
            'longitude': place['geometry']['location'].get('lng')
        })
    
    return filtered_places

# Get the data for assurance companies without limit
assurance_companies = get_assurance_companies(assurance_companies_names, city_coords)

# Convert to DataFrame for better readability
df = pd.DataFrame(assurance_companies)

# Save the DataFrame to a CSV file
df.to_csv('Assurance_Companies_Data.csv', index=False)