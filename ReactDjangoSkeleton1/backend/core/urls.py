from django.urls import path
from core.views import (
    general_core, detail_core
)

urlpatterns = [
    path('', general_core, name="core"),
    path('<int:element_id>/', detail_core, name="core_detail"),
]
