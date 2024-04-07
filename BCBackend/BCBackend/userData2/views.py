from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from django.conf import settings

from .models import *
from rest_framework.response import Response


class UserFormView(APIView):
    def post(self, request):
        pass

class TimelineView(APIView):
    def get(self, request):
        output = {
            'jobs':[{
                'job': "Software Eng",
                "skills":["python", "html", "react", "javascript"]
            },
                {
                    'job': "Flouriest",
                    'skills':['cutting', 'watering', "growing"]
                }]
        }
        return Response(output)

