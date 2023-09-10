from django.urls import path
from tags.views import (
    general_tag,
    general_tag_class,
    general_tag_preset
)

urlpatterns = [
    path('class/', general_tag_class, name="tag_class"),
    path('', general_tag, name="tag"),
    path('preset/', general_tag_preset, name="tag_preset"),
]
