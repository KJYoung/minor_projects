from django.urls import path
from todos.views import general_todo, toggle_todo, general_todo_category, detail_todo_category

urlpatterns = [
    path('', general_todo, name="todo"),
    # Params: year<Number>, month<Number>
    path('toggle/<int:todo_id>/', toggle_todo, name="todo_toggle"),
    path('category/', general_todo_category, name="todo_category"),
    path('category/<int:categ_id>/', detail_todo_category, name="todo_category_detail"),
]
