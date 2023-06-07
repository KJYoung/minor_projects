import json

from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from core.models import Core

from json.decoder import JSONDecodeError


@require_http_methods(['GET', 'POST'])
def general_core(request):
    """
    GET : get element list
    POST : create element
    """
    if request.method == 'GET':
        result = []
        for core_element in Core.objects.all():
            result.append(
                {
                    "id": core_element.id,
                    "name": core_element.name,
                }
            )
        return JsonResponse({"elements": result}, safe=False)
    else:  ## post
        try:
            req_data = json.loads(request.body.decode())

            element = Core(
                name=req_data["name"],
            )
            element.save()
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id}, status=201)
