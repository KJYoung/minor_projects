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
    print("Called")
    if request.method == 'GET':
        result = []
        for gr_obj in Core.objects.all():
            result.append(
                {
                    "id": gr_obj.id,
                    "name": gr_obj.group_name,
                }
            )
        return JsonResponse({"elements": result}, safe=False)
    else:  ## post
        try:
            req_data = json.loads(request.body.decode())
           
            group = Core(
                name=req_data["name"],
            )
            group.save()
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": group.id}, status=201)
