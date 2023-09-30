from django.urls import path
from tags.views import (
    general_tag,
    general_tag_class,
    general_tag_preset,
    tag_detail,
    tag_preset_detail,
)

urlpatterns = [
    path('class/', general_tag_class, name="tag_class"),
    path('', general_tag, name="tag"),
    path('<int:tag_id>/', tag_detail, name="tag_detail"),
    path('preset/', general_tag_preset, name="tag_preset"),
    path('preset/<int:tag_preset_id>/', tag_preset_detail, name="tag_preset_detail"),
]
