# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock Hotel Booking Agent for demo purposes."""

import random
from datetime import datetime

from google.adk import Agent
from google.genai import types


async def search_hotels(
    location: str,
    checkin_date: str,
    checkout_date: str,
    guests: int = 2,
    room_type: str = "any",
) -> str:
    """Search for available hotels.

    Args:
        location: Location/city to search hotels.
        checkin_date: Check-in date (YYYY-MM-DD).
        checkout_date: Check-out date (YYYY-MM-DD).
        guests: Number of guests.
        room_type: Type of room (standard, deluxe, suite, any).

    Returns:
        JSON string with available hotels.
    """
    import json

    # Mock hotel options
    hotels = [
        {
            "hotel_id": f"HT{random.randint(1000, 9999)}",
            "hotel_name": "Ocean View Resort",
            "location": location,
            "rating": 4.5,
            "room_type": "Deluxe Ocean View",
            "price_per_night": random.randint(8000, 25000),
            "currency": "JPY",
            "amenities": ["Free WiFi", "Pool", "Restaurant", "Beach Access"],
            "rooms_available": random.randint(3, 20),
        },
        {
            "hotel_id": f"HT{random.randint(1000, 9999)}",
            "hotel_name": "City Grand Hotel",
            "location": location,
            "rating": 4.2,
            "room_type": "Standard Twin",
            "price_per_night": random.randint(6000, 18000),
            "currency": "JPY",
            "amenities": ["Free WiFi", "Breakfast Included", "Gym", "Parking"],
            "rooms_available": random.randint(3, 20),
        },
        {
            "hotel_id": f"HT{random.randint(1000, 9999)}",
            "hotel_name": "Luxury Beach Suite Hotel",
            "location": location,
            "rating": 4.8,
            "room_type": "Presidential Suite",
            "price_per_night": random.randint(30000, 60000),
            "currency": "JPY",
            "amenities": ["Free WiFi", "Private Pool", "Spa", "Concierge", "Beach Access"],
            "rooms_available": random.randint(1, 5),
        },
    ]

    result = {
        "search_criteria": {
            "location": location,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "guests": guests,
            "room_type": room_type,
        },
        "hotels": hotels,
        "total_results": len(hotels),
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


async def book_hotel(
    location: str = "",
    hotel_id: str = "",
    guest_name: str = "",
    guest_email: str = "",
    checkin_date: str = "",
    checkout_date: str = "",
    guests: int = 2,
    room_type: str = "any",
    special_requests: str = "",
) -> str:
    """Book a hotel room.

    Args:
        location: Location/city for the hotel (optional, for searching).
        hotel_id: The hotel ID to book (optional, will be generated if not provided).
        guest_name: Name of the guest.
        guest_email: Email of the guest.
        checkin_date: Check-in date (YYYY-MM-DD).
        checkout_date: Check-out date (YYYY-MM-DD).
        guests: Number of guests.
        room_type: Type of room (standard, twin, deluxe, suite, any).
        special_requests: Any special requests.

    Returns:
        JSON string with booking confirmation.
    """
    import json
    from datetime import datetime as dt

    # Mock booking confirmation - generate hotel_id if not provided
    if not hotel_id:
        hotel_id = f"HT{random.randint(1000, 9999)}"

    booking_id = f"HB{random.randint(100000, 999999)}"
    confirmation_code = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))}"

    # Calculate number of nights
    try:
        checkin = dt.strptime(checkin_date, "%Y-%m-%d")
        checkout = dt.strptime(checkout_date, "%Y-%m-%d")
        nights = (checkout - checkin).days
    except:
        nights = 3  # Fallback to default

    # Select hotel name and price based on room type
    hotel_options = {
        "deluxe": ("Ocean View Resort", random.randint(12000, 25000)),
        "suite": ("Luxury Beach Suite Hotel", random.randint(30000, 60000)),
        "twin": ("City Grand Hotel", random.randint(8000, 18000)),
        "standard": ("City Grand Hotel", random.randint(6000, 15000)),
    }

    # Default hotel
    hotel_name = "Naha City Hotel"
    price_per_night = random.randint(8000, 18000)

    # Override if room_type matches
    for key, (name, price) in hotel_options.items():
        if key in room_type.lower():
            hotel_name = name
            price_per_night = price
            break

    booking = {
        "status": "confirmed",
        "booking_id": booking_id,
        "confirmation_code": confirmation_code,
        "hotel_id": hotel_id,
        "hotel_name": hotel_name,
        "location": location if location else "Naha, Okinawa",
        "guest_name": guest_name,
        "guest_email": guest_email,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "number_of_guests": guests,
        "number_of_nights": nights,
        "room_type": room_type if room_type != "any" else "Standard Twin",
        "price_per_night": price_per_night,
        "total_price": price_per_night * nights,
        "currency": "JPY",
        "special_requests": special_requests if special_requests else "None",
        "booking_date": datetime.now().isoformat(),
        "message": f"Hotel successfully booked! Confirmation code: {confirmation_code}. We look forward to your stay.",
    }

    return json.dumps(booking, indent=2, ensure_ascii=False)


async def cancel_hotel_booking(booking_id: str) -> str:
    """Cancel a hotel booking.

    Args:
        booking_id: The booking ID to cancel.

    Returns:
        JSON string with cancellation confirmation.
    """
    import json

    result = {
        "status": "cancelled",
        "booking_id": booking_id,
        "cancellation_date": datetime.now().isoformat(),
        "refund_amount": random.randint(18000, 90000),
        "currency": "JPY",
        "cancellation_fee": 0,
        "message": "Hotel booking successfully cancelled. Full refund will be processed within 5-7 business days.",
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


async def get_hotel_details(hotel_id: str) -> str:
    """Get detailed information about a hotel.

    Args:
        hotel_id: The hotel ID.

    Returns:
        JSON string with hotel details.
    """
    import json

    details = {
        "hotel_id": hotel_id,
        "hotel_name": "Ocean View Resort",
        "location": "Naha, Okinawa",
        "address": "1-2-3 Beach Street, Naha City, Okinawa",
        "rating": 4.5,
        "description": "Luxury beachfront resort with stunning ocean views and world-class amenities.",
        "amenities": ["Free WiFi", "Pool", "Restaurant", "Beach Access", "Spa", "Gym"],
        "checkin_time": "15:00",
        "checkout_time": "11:00",
        "contact_phone": "+81-98-XXX-XXXX",
        "contact_email": "info@oceanviewresort.com",
    }

    return json.dumps(details, indent=2, ensure_ascii=False)


root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='hotel_booking_agent',
    description=(
        'Hotel booking agent that can search for hotels and make reservations. '
        'Provides hotel search, booking, cancellation, and information services.'
    ),
    instruction="""
You are a hotel booking assistant.

Your capabilities:
1. Search for available hotels in any location
2. Book hotel rooms for guests (even with minimal information)
3. Cancel existing hotel bookings
4. Provide detailed hotel information

When helping users:
- For booking requests: If you receive a booking request with location, dates, and guest information, proceed with the booking immediately using the book_hotel function
- You can make reasonable assumptions (e.g., twin room if not specified, standard amenities)
- Always provide booking confirmation with confirmation code
- For search requests: Use search_hotels to find available options
- Handle special requests professionally

Important:
- If the request is clearly to make a booking (not just search), use book_hotel directly
- Collect essential info: location, checkin_date, checkout_date, guests
- Generate appropriate hotel details based on the provided information

Always be helpful and process booking requests efficiently.
""",
    tools=[
        search_hotels,
        book_hotel,
        cancel_hotel_booking,
        get_hotel_details,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ],
    ),
)
