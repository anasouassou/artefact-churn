<<<<<<< HEAD
import os
import openai
from flask import Flask, request, render_template, send_file
import googlemaps
import pandas as pd
import math
import folium
from io import BytesIO

app = Flask(__name__)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Load API keys from environment variables
api_key = os.getenv('AZURE_OPENAI_API_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

if not api_key or not endpoint or not google_maps_api_key:
    raise ValueError("API key, endpoint, and Google Maps API key must be set in environment variables.")

openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = endpoint
openai.api_version = "2023-05-15"

gmaps = googlemaps.Client(key=google_maps_api_key)

def extract_info_from_text_gpt(text):
    prompt = f"""
        Extract the following details from the text, make sure to delete any word that's like 'insurance', 
        'assurance' from the extracted values, and for all extracted values remove the words that are not significant:
        - Assurance company name
        - Address
        - Radius in km

        Text: {text}

        Output format:
        Company Name: <company_name>
        Address: <address>
        Radius: <radius>
        """
    
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # Replace with your actual deployment name
        messages=[
            {"role": "system", "content": "You are an assistant that extracts specific information from text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    response_text = response.choices[0].message['content'].strip()
    
    # Extract company name, address, and radius from the response
    company_name = None
    address = None
    radius = None

    for line in response_text.split('\n'):
        if line.startswith("Company Name:"):
            company_name = line.split(":")[1].strip()
        elif line.startswith("Address:"):
            address = line.split(":")[1].strip()
        elif line.startswith("Radius:"):
            radius = float(line.split(":")[1].strip().replace('km', '').strip())
    
    return company_name, address, radius

def geocode_address(address):
    geocode_result = gmaps.geocode(address)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        return location['lat'], location['lng']
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def find_nearest_assurance_company(company_name, client_address, search_radius):
    client_lat, client_lng = geocode_address(client_address)
    if client_lat is None or client_lng is None:
        return "Client address could not be geocoded.", [], [], 0, None, None

    file_path = r'data\final_data.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Find the nearest company matching the given searched_name
    assurance_companies = df[df['company'].str.contains(company_name, case=False, na=False)]
    
    if assurance_companies.empty:
        return "No matching assurance companies found in the database.", [], [], 0, None, None

    nearest_company = None
    shortest_distance = float('inf')

    for _, company in assurance_companies.iterrows():
        company_lat = company['latitude']
        company_lng = company['longitude']
        distance = haversine(client_lat, client_lng, company_lat, company_lng)
        if distance < shortest_distance:
            shortest_distance = distance
            nearest_company = company

    if nearest_company is not None:
        nearest_result = f"Nearest Assurance Company: {nearest_company['name']}, Address: {nearest_company['address']}, Distance: {shortest_distance:.2f} km"
    else:
        nearest_result = "No assurance company found."

    # Find all assurance companies within the specified radius
    surrounding_companies = []
    unique_companies = set()
    surrounding_count = 0

    for _, company in df.iterrows():
        company_lat = company['latitude']
        company_lng = company['longitude']
        distance = haversine(client_lat, client_lng, company_lat, company_lng)
        if distance <= search_radius:
            company_info = (company['name'], company['address'], company_lat, company_lng)
            if company_info not in unique_companies:
                unique_companies.add(company_info)
                surrounding_companies.append({
                    'name': company['name'],
                    'address': company['address'],
                    'company' : company['company'],
                    'distance': distance,
                    'lat': company_lat,
                    'lng': company_lng
                })
                surrounding_count += 1

    # Sort the surrounding companies by distance
    surrounding_companies.sort(key=lambda x: x['distance'])

    return nearest_result, nearest_company, surrounding_companies, surrounding_count, client_lat, client_lng

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        company_name, client_address, search_radius = extract_info_from_text_gpt(user_input)
        
        if company_name and client_address and search_radius is not None:
            nearest_result, nearest_company, surrounding_results, surrounding_count, client_lat, client_lng = find_nearest_assurance_company(
                company_name, client_address, search_radius
            )

            # Group surrounding companies by their name
            company_name_counts = {}
            for company in surrounding_results:
                company_name = company['company']
                if company_name in company_name_counts:
                    company_name_counts[company_name] += 1
                else:
                    company_name_counts[company_name] = 1


            # Create a DataFrame for grouped company name counts and sort by count
            grouped_counts_df = pd.DataFrame(list(company_name_counts.items()), columns=['Company Name', 'Count']).sort_values(by='Count', ascending=False)

            # Convert the surrounding results to a DataFrame
            surrounding_df = pd.DataFrame(surrounding_results)
            surrounding_csv = surrounding_df.to_csv(index=False)
            csv_io = BytesIO(surrounding_csv.encode())

            # Create a map centered around the client's address
            map_center = [client_lat, client_lng]
            m = folium.Map(location=map_center, zoom_start=13)
            
            # Add a circle representing the search radius
            folium.Circle(
                radius=search_radius * 1000,
                location=map_center,
                color="blue",
                fill=True,
                fill_color="blue",
            ).add_to(m)
            
            # Add a marker for the client's address
            folium.Marker(
                location=map_center,
                tooltip="Client Address",
                icon=folium.Icon(color="red")
            ).add_to(m)
            
            # Add markers for the surrounding companies
            for company in surrounding_results:
                folium.Marker(
                    location=[company['lat'], company['lng']],
                    tooltip=f"{company['company']}\n- {company['address']}\n- Distance: {company['distance']:.2f} km"
                ).add_to(m)
            
            map_html = m._repr_html_()

            return render_template('index.html', 
                                   nearest_result=nearest_result, 
                                   surrounding_count=surrounding_count, 
                                   grouped_counts=grouped_counts_df.to_html(classes='data', header=True, index=False), 
                                   map_html=map_html,
                                   surrounding_csv=csv_io.getvalue().decode())

    return render_template('index.html')

@app.route('/download')
def download_file():
    surrounding_csv = request.args.get('surrounding_csv')
    if surrounding_csv:
        return send_file(BytesIO(surrounding_csv.encode()),
                         mimetype='text/csv',
                         download_name='surrounding_assurance_companies.csv',
                         as_attachment=True)
    return "No file to download."

if __name__ == '__main__':
    app.run(debug=True)
=======
import os
import openai
from flask import Flask, request, render_template, send_file
import googlemaps
import pandas as pd
import math
import folium
from io import BytesIO

app = Flask(__name__)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Load API keys from environment variables
api_key = os.getenv('AZURE_OPENAI_API_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

if not api_key or not endpoint or not google_maps_api_key:
    raise ValueError("API key, endpoint, and Google Maps API key must be set in environment variables.")

openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = endpoint
openai.api_version = "2023-05-15"

gmaps = googlemaps.Client(key=google_maps_api_key)

def extract_info_from_text_gpt(text):
    prompt = f"""
        Extract the following details from the text, make sure to delete any word that's like 'insurance', 
        'assurance' from the extracted values, and for all extracted values remove the words that are not significant:
        - Assurance company name
        - Address
        - Radius in km

        Text: {text}

        Output format:
        Company Name: <company_name>
        Address: <address>
        Radius: <radius>
        """
    
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # Replace with your actual deployment name
        messages=[
            {"role": "system", "content": "You are an assistant that extracts specific information from text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    response_text = response.choices[0].message['content'].strip()
    
    # Extract company name, address, and radius from the response
    company_name = None
    address = None
    radius = None

    for line in response_text.split('\n'):
        if line.startswith("Company Name:"):
            company_name = line.split(":")[1].strip()
        elif line.startswith("Address:"):
            address = line.split(":")[1].strip()
        elif line.startswith("Radius:"):
            radius = float(line.split(":")[1].strip().replace('km', '').strip())
    
    return company_name, address, radius

def geocode_address(address):
    geocode_result = gmaps.geocode(address)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        return location['lat'], location['lng']
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def find_nearest_assurance_company(company_name, client_address, search_radius):
    client_lat, client_lng = geocode_address(client_address)
    if client_lat is None or client_lng is None:
        return "Client address could not be geocoded.", [], [], 0, None, None

    file_path = r'Final_data.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Find the nearest company matching the given searched_name
    assurance_companies = df[df['company'].str.contains(company_name, case=False, na=False)]
    
    if assurance_companies.empty:
        return "No matching assurance companies found in the database.", [], [], 0, None, None

    nearest_company = None
    shortest_distance = float('inf')

    for _, company in assurance_companies.iterrows():
        company_lat = company['latitude']
        company_lng = company['longitude']
        distance = haversine(client_lat, client_lng, company_lat, company_lng)
        if distance < shortest_distance:
            shortest_distance = distance
            nearest_company = company

    if nearest_company is not None:
        nearest_result = f"Nearest Assurance Company: {nearest_company['name']}, Address: {nearest_company['address']}, Distance: {shortest_distance:.2f} km"
    else:
        nearest_result = "No assurance company found."

    # Find all assurance companies within the specified radius
    surrounding_companies = []
    unique_companies = set()
    surrounding_count = 0

    for _, company in df.iterrows():
        company_lat = company['latitude']
        company_lng = company['longitude']
        distance = haversine(client_lat, client_lng, company_lat, company_lng)
        if distance <= search_radius:
            company_info = (company['name'], company['address'], company_lat, company_lng)
            if company_info not in unique_companies:
                unique_companies.add(company_info)
                surrounding_companies.append({
                    'name': company['name'],
                    'address': company['address'],
                    'company' : company['company'],
                    'distance': distance,
                    'lat': company_lat,
                    'lng': company_lng
                })
                surrounding_count += 1

    # Sort the surrounding companies by distance
    surrounding_companies.sort(key=lambda x: x['distance'])

    return nearest_result, nearest_company, surrounding_companies, surrounding_count, client_lat, client_lng

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        company_name, client_address, search_radius = extract_info_from_text_gpt(user_input)
        
        if company_name and client_address and search_radius is not None:
            nearest_result, nearest_company, surrounding_results, surrounding_count, client_lat, client_lng = find_nearest_assurance_company(
                company_name, client_address, search_radius
            )

            # Group surrounding companies by their name
            company_name_counts = {}
            for company in surrounding_results:
                company_name = company['company']
                if company_name in company_name_counts:
                    company_name_counts[company_name] += 1
                else:
                    company_name_counts[company_name] = 1


            # Create a DataFrame for grouped company name counts and sort by count
            grouped_counts_df = pd.DataFrame(list(company_name_counts.items()), columns=['Company Name', 'Count']).sort_values(by='Count', ascending=False)

            # Convert the surrounding results to a DataFrame
            surrounding_df = pd.DataFrame(surrounding_results)
            surrounding_csv = surrounding_df.to_csv(index=False)
            csv_io = BytesIO(surrounding_csv.encode())

            # Create a map centered around the client's address
            map_center = [client_lat, client_lng]
            m = folium.Map(location=map_center, zoom_start=13)
            
            # Add a circle representing the search radius
            folium.Circle(
                radius=search_radius * 1000,
                location=map_center,
                color="blue",
                fill=True,
                fill_color="blue",
            ).add_to(m)
            
            # Add a marker for the client's address
            folium.Marker(
                location=map_center,
                tooltip="Client Address",
                icon=folium.Icon(color="red")
            ).add_to(m)
            
            # Add markers for the surrounding companies
            for company in surrounding_results:
                folium.Marker(
                    location=[company['lat'], company['lng']],
                    tooltip=f"{company['company']}\n- {company['address']}\n- Distance: {company['distance']:.2f} km"
                ).add_to(m)
            
            map_html = m._repr_html_()

            return render_template('index.html', 
                                   nearest_result=nearest_result, 
                                   surrounding_count=surrounding_count, 
                                   grouped_counts=grouped_counts_df.to_html(classes='data', header=True, index=False), 
                                   map_html=map_html,
                                   surrounding_csv=csv_io.getvalue().decode())

    return render_template('index.html')

@app.route('/download')
def download_file():
    surrounding_csv = request.args.get('surrounding_csv')
    if surrounding_csv:
        return send_file(BytesIO(surrounding_csv.encode()),
                         mimetype='text/csv',
                         download_name='surrounding_assurance_companies.csv',
                         as_attachment=True)
    return "No file to download."

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 57f5436bd44cbb7a53adf53338d340eb8c5d4bcb
