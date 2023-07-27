from django.urls import path
from todos.views import (
    general_todo,
)

urlpatterns = [
    path('', general_todo, name="todo"),
    # Params: year<Number>, month<Number>
]
