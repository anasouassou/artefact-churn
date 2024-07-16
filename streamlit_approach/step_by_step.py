import openai
import streamlit as st
import os
from dotenv import load_dotenv
import googlemaps
import pandas as pd
import math
import folium
from streamlit_folium import st_folium

# Load environment variables from .env file
load_dotenv()

# Load API keys from environment variables or .env file
api_key = os.getenv('AZURE_OPENAI_API_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

if not api_key or not endpoint or not google_maps_api_key:
    st.error("API key, endpoint, and Google Maps API key must be set in environment variables.")
else:
    openai.api_type = "azure"
    openai.api_key = api_key
    openai.api_base = endpoint
    openai.api_version = "2023-05-15"

    gmaps = googlemaps.Client(key=google_maps_api_key)

    def extract_info_from_text_gpt(text):
        prompt = f"""
        Extract the following details from the text, make sure to delete any word that's like 'insurance', 'assurance' from the extracted values, and for all extracted values remove the words that are not significant:
        - Assurance company name
        - Address
        - Radius in km

        Text: {text}

        Output format:
        Company Name: <company_name>
        Address: <address>
        Radius: <radius>
        """
        
        try:
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
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None, None, None

    def geocode_address(address):
        try:
            geocode_result = gmaps.geocode(address)
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                return location['lat'], location['lng']
            else:
                return None, None
        except Exception as e:
            st.write(f"An error occurred: {e}")
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

        file_path = r'final_data.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Find the nearest company matching the given company name
        assurance_companies = df[df['Name'].str.contains(company_name, case=False, na=False)]
        
        if assurance_companies.empty:
            return "No matching assurance companies found in the database.", [], [], 0, None, None

        nearest_company = None
        shortest_distance = float('inf')

        for _, company in assurance_companies.iterrows():
            company_lat = company['Latitude']
            company_lng = company['Longitude']
            distance = haversine(client_lat, client_lng, company_lat, company_lng)
            if distance < shortest_distance:
                shortest_distance = distance
                nearest_company = company

        if nearest_company is not None:
            nearest_result = f"Nearest Assurance Company: {nearest_company['Name']}, Address: {nearest_company['Address']}, Distance: {shortest_distance:.2f} km"
        else:
            nearest_result = "No assurance company found."

        # Find all assurance companies within the specified radius
        surrounding_companies = []
        unique_companies = set()
        surrounding_count = 0

        for _, company in df.iterrows():
            company_lat = company['Latitude']
            company_lng = company['Longitude']
            distance = haversine(client_lat, client_lng, company_lat, company_lng)
            if distance <= search_radius:
                company_info = (company['Name'], company['Address'], company_lat, company_lng)
                if company_info not in unique_companies:
                    unique_companies.add(company_info)
                    surrounding_companies.append({
                        'name': company['Name'],
                        'address': company['Address'],
                        'distance': distance,
                        'lat': company_lat,
                        'lng': company_lng
                    })
                    surrounding_count += 1

        # Sort the surrounding companies by distance
        surrounding_companies.sort(key=lambda x: x['distance'])

        return nearest_result, nearest_company, surrounding_companies, surrounding_count, client_lat, client_lng

    # Streamlit app
    st.title("Assurance Company Churn Predictor")

    if 'company_name' not in st.session_state:
        st.session_state.company_name = ""
    if 'client_address' not in st.session_state:
        st.session_state.client_address = ""
    if 'search_radius' not in st.session_state:
        st.session_state.search_radius = 0.0

    if 'nearest_result' not in st.session_state:
        st.session_state.nearest_result = ""
    if 'surrounding_results' not in st.session_state:
        st.session_state.surrounding_results = []
    if 'surrounding_count' not in st.session_state:
        st.session_state.surrounding_count = 0
    if 'client_lat' not in st.session_state:
        st.session_state.client_lat = 0
    if 'client_lng' not in st.session_state:
        st.session_state.client_lng = 0
    if 'show_map' not in st.session_state:
        st.session_state.show_map = False

    user_input = st.text_area('Enter the details (e.g., "we are WAFA, what are the nearest assurance companies from 55, Bd Abdelmoumen, Casablanca, Grand Casablanca 20029 within a radius of 5km"):')

    if st.button('Extract Information'):
        if user_input:
            company_name, client_address, search_radius = extract_info_from_text_gpt(user_input)
            
            if company_name and client_address and search_radius is not None:
                st.session_state.company_name = company_name
                st.session_state.client_address = client_address
                st.session_state.search_radius = search_radius
                
                st.write(f"**Company Name:** {company_name}")
                st.write(f"**Client Address:** {client_address}")
                st.write(f"**Search Radius:** {search_radius} km")
            else:
                st.write("Could not extract information. Please check the input format.")
        else:
            st.write("Please enter the details.")

    if st.button('Find Nearest Assurance Companies'):
        if st.session_state.company_name and st.session_state.client_address and st.session_state.search_radius:
            nearest_result, nearest_company, surrounding_results, surrounding_count, client_lat, client_lng = find_nearest_assurance_company(
                st.session_state.company_name, st.session_state.client_address, st.session_state.search_radius
            )

            st.session_state.nearest_result = nearest_result
            st.session_state.surrounding_results = surrounding_results
            st.session_state.surrounding_count = surrounding_count
            st.session_state.client_lat = client_lat
            st.session_state.client_lng = client_lng
            st.session_state.show_map = False  # Reset map visibility

    if st.session_state.nearest_result:
        st.markdown(f'<p>Nearest Assurance Company: {st.session_state.nearest_result}</p>', unsafe_allow_html=True)

    if st.session_state.surrounding_results:
        st.markdown(f'<p>Surrounding Assurance Companies (within {st.session_state.search_radius} km): {st.session_state.surrounding_count}</p>', unsafe_allow_html=True)

        # Group surrounding companies by the first word of their name
        company_name_counts = {}
        for company in st.session_state.surrounding_results:
            first_word = company['name'].split()[0]
            if first_word in company_name_counts:
                company_name_counts[first_word] += 1
            else:
                company_name_counts[first_word] = 1

        # Create a DataFrame for grouped company name counts and sort by count
        grouped_counts_df = pd.DataFrame(list(company_name_counts.items()), columns=['Company Name', 'Count'])
        grouped_counts_df = grouped_counts_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        grouped_counts_df.index += 1



        # Display the grouped counts as a table without the index
        st.write("### Grouped by Company Name:")
        st.dataframe(grouped_counts_df, width=300, height=400,)

        # Convert the surrounding results to a DataFrame
        surrounding_df = pd.DataFrame(st.session_state.surrounding_results)

        # Provide a download button for the DataFrame
        csv = surrounding_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data for surrounding companies",
            data=csv,
            file_name='surrounding_assurance_companies.csv',
            mime='text/csv',
        )

        if st.button("Show Map"):
            st.session_state.show_map = True

        if st.session_state.show_map:
            # Create a map centered around the client's address
            map_center = [st.session_state.client_lat, st.session_state.client_lng]
            m = folium.Map(location=map_center, zoom_start=13)
            
            # Add a circle representing the search radius
            folium.Circle(
                radius=st.session_state.search_radius * 1000,
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
            for company in st.session_state.surrounding_results:
                folium.Marker(
                    location=[company['lat'], company['lng']],
                    tooltip=f"{company['name']}\n{company['address']}\nDistance: {company['distance']:.2f} km", 
                    icon=folium.Icon(color="blue")
                ).add_to(m)
            
            # Display the map in Streamlit
            st_folium(m, width=700, height=500)
