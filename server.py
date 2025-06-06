import os
import json
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


load_dotenv()


def get_access_token(sandbox: bool = True) -> Dict[str, Any]:
   
    client_id = os.getenv("AMADEUS_CLIENT_ID")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
    base_url = "https://test.api.amadeus.com" if sandbox else "https://api.amadeus.com"
    token_url = f"{base_url}/v1/security/oauth2/token"

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    try:
        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()
        return {
            "access_token": token_data.get("access_token"),
            "expires_in": token_data.get("expires_in"),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Error getting token: {e}"}


def search_hotels_by_city(access_token: str, city_code: str) -> Dict[str, Any]:
   
    base_url = "https://test.api.amadeus.com"
    endpoint = f"{base_url}/v1/reference-data/locations/hotels/by-city"

    headers = {"Authorization": f"Bearer {access_token}"}

    params = {"cityCode": city_code}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error searching for hotels: {e}"}


def get_hotel_offers(access_token: str, hotel_ids: List[str]) -> Dict[str, Any]:
    
    base_url = "https://test.api.amadeus.com"
    endpoint = f"{base_url}/v3/shopping/hotel-offers"

    headers = {"Authorization": f"Bearer {access_token}"}

    params = {"hotelIds": ",".join(hotel_ids)}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error getting hotel offers: {e}"}


def search_flight_offers(
    access_token: str,
    origin: str,
    destination: str,
    departure_date: str,
    adults: int,
    return_date: Optional[str] = None,
    currency: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches for flight offers using the /v2/shopping/flight-offers endpoint.

    Parameters:
        access_token (str): A valid OAuth 2.0 token.
        origin (str): IATA code for the origin airport (e.g., "BOS").
        destination (str): IATA code for the destination airport (e.g., "PAR").
        departure_date (str): Departure date in "YYYY-MM-DD" format.
        adults (int): Number of adult passengers (≥12 years old).
        return_date (str | None): Return date in "YYYY-MM-DD" format (optional).
        currency (str | None): ISO currency code (optional).

    Returns:
        dict: JSON with the flight offers, or {"error": "..."} on failure.
    """
    base_url = "https://test.api.amadeus.com"
    endpoint = f"{base_url}/v2/shopping/flight-offers"

    headers = {"Authorization": f"Bearer {access_token}"}

    params: Dict[str, Any] = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "adults": adults,
    }
    if return_date:
        params["returnDate"] = return_date
    if currency:
        params["currencyCode"] = currency

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error getting flight offers: {e}"}


def search_pois(lat: float, lon: float, access_token: str):
    """
    Gets points of interest (restaurants, activities, etc.) near a given point by latitude and longitude.

    """
    base_url = "https://test.api.amadeus.com"
    endpoint = f"{base_url}/v1/shopping/activities"

    headers = {"Authorization": f"Bearer {access_token}"}

    params = {"latitude": lat, "longitude": lon}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error getting points of interest: {e}"}


mcp = FastMCP("TravelToolsServer")


@mcp.tool()
def tool_flight_offer(
    origin: str, destination: str, departure_date: str, return_date: str, adults: str
) -> str:
    """
    Searches for flight offers for the given parameters

    Parameters (all as strings):
      - origin (str): IATA code for the origin airport (e.g., "BOS").
      - destination (str): IATA code for the destination airport (e.g., "PAR").
      - departure_date (str): Departure date "YYYY-MM-DD".
      - return_date (str): Return date "YYYY-MM-DD" or "none" for a one-way trip.
      - adults (str): Number of adults; will be converted to int.

    Returns:
      - str: one offer for a flight
    """

    try:
        adults_int = int(adults)
    except ValueError:
        return "Error: 'adults' must be an integer."

    if return_date.lower() in ("none", ""):
        return_date_val: Optional[str] = None
    else:
        return_date_val = return_date

    token_info = get_access_token(sandbox=True)
    if "error" in token_info:
        return token_info["error"]

    access_token = token_info["access_token"]

    offers = search_flight_offers(
        access_token=access_token,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        adults=adults_int,
        return_date=return_date_val,
    )
    if "error" in offers:
        return offers["error"]

    data_list: List[Any] = offers.get("data", [])
    if not data_list:
        return "No flight offers found for those parameters."

    first_offer = data_list[0]
    return json.dumps(first_offer, indent=2)


@mcp.tool()
def get_pois(latitude: float, longitude: float):
    """
    Gets points of interest (restaurants, activities, etc.) near a given point by latitude and longitude.

    Path parameters:
      - latitude (float): latitude for the coordinates where you want to search for POIs.
      - longitude (float): longitude for the coordinates where you want to search for POIs.

    Returns:
      - 5 points of interes based on the given coordinate
    """

    token_res = get_access_token(sandbox=True)
    if "error" in token_res:
        return {"error", token_res["error"]}

    access_token = token_res["access_token"]

    pois = search_pois(latitude, longitude, access_token)

    if "error" in pois:
        return {"error", token_res["error"]}

    data: List[Any] = pois.get("data", [])

    if not data:
        return {
            "error",
            "No points of interest were found near the desired coordinates",
        }

    pretty_json = json.dumps(data[:4], indent=2)
    return pretty_json


@mcp.tool()
def tool_search_hotels_by_city(city_code: str) -> Dict[str, Any]:
    """
    MCP tool that searches for hotels in the city {city_code}, takes the first 5 results,
    and gets their offers.

    Parameters:
        city_code (str): IATA code for the city (e.g., "PAR", "MAD").
    Returns:
        dict: JSON with the offers for the first 5 hotels found.
    """

    token_info = get_access_token(sandbox=True)
    if "error" in token_info:
        return {"error": token_info["error"]}

    access_token = token_info["access_token"]

    hotels_response = search_hotels_by_city(access_token, city_code)
    if "error" in hotels_response:
        return {"error": hotels_response["error"]}

    hotels_data = hotels_response.get("data", [])
    first_five_ids = []
    for entry in hotels_data[:5]:

        hotel_id = entry.get("hotelId")
        if hotel_id:
            first_five_ids.append(hotel_id)

    if not first_five_ids:
        return {"error": "No hotels were found for the specified city."}

    offers_response = get_hotel_offers(access_token, first_five_ids)
    return offers_response


if __name__ == "__main__":
    mcp.run(transport="stdio")