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
    hotel_id: str,
    guest_name: str,
    guest_email: str,
    checkin_date: str,
    checkout_date: str,
    guests: int = 2,
    special_requests: str = "",
) -> str:
    """Book a hotel room.

    Args:
        hotel_id: The hotel ID to book.
        guest_name: Name of the guest.
        guest_email: Email of the guest.
        checkin_date: Check-in date (YYYY-MM-DD).
        checkout_date: Check-out date (YYYY-MM-DD).
        guests: Number of guests.
        special_requests: Any special requests.

    Returns:
        JSON string with booking confirmation.
    """
    import json

    # Mock booking confirmation
    booking_id = f"HB{random.randint(100000, 999999)}"
    confirmation_code = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))}"

    # Calculate number of nights (simplified)
    nights = 3  # Mock value

    booking = {
        "status": "confirmed",
        "booking_id": booking_id,
        "confirmation_code": confirmation_code,
        "hotel_id": hotel_id,
        "guest_name": guest_name,
        "guest_email": guest_email,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "number_of_guests": guests,
        "number_of_nights": nights,
        "total_price": random.randint(6000, 30000) * nights,
        "currency": "JPY",
        "special_requests": special_requests if special_requests else "None",
        "booking_date": datetime.now().isoformat(),
        "message": f"Hotel successfully booked! Confirmation code: {confirmation_code}",
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
2. Book hotel rooms for guests
3. Cancel existing hotel bookings
4. Provide detailed hotel information

When helping users:
- Ask for necessary information (location, dates, number of guests, preferences)
- Provide clear hotel options with prices and amenities
- Confirm bookings with confirmation codes
- Handle special requests professionally
- Provide hotel details when requested

Always be helpful and ensure all required information is collected before making bookings.
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
