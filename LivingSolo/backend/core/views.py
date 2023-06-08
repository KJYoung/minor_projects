import json

from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
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
        return JsonResponse({"id": element.id, "name": element.name}, status=201)


@require_http_methods(['PUT', 'DELETE'])
def detail_core(request, element_id):
    """
    PUT : edit element's content
    DELETE : delete element
    """
    if request.method == 'PUT':
        try:
            data = json.loads(request.body.decode())
            core_id = int(element_id)
            core_obj = Core.objects.get(pk=core_id)

            core_obj.name = data["name"]
            core_obj.save()
            return JsonResponse({"message": "success"}, status=200)
        except Core.DoesNotExist:
            return HttpResponseNotFound()
        except Exception:
            return HttpResponseBadRequest()
    else:  ## delete
        try:
            core_id = int(element_id)
            core_obj = Core.objects.get(pk=core_id)

            core_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except Core.DoesNotExist:
            return HttpResponseNotFound()
        except Exception:
            return HttpResponseBadRequest()
