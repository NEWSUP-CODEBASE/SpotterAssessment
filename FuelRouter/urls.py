from django.urls import path
from . import views
from .views import FuelRouteAPIView, fuel_route_get, spotter_fuel_view

urlpatterns = [
    path('spotter-fuel-dashboard/', views.spotter_fuel_view, name='spotter-fuel-ui'),
    path('get-route/', views.get_route, name='get_route_api'),
    path('route/', FuelRouteAPIView.as_view(), name='fuel-route-api'),
    path('route/simple/', fuel_route_get, name='fuel-route-get'),
]