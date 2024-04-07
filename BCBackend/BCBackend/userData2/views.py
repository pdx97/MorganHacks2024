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
        data = request.data
        fields = data["data"]["fields"]
        formatedData = {}
        for field in fields:
            print("{")
            for key in field.keys():
                if key == "label" and field["label"] == "First Name":
                    formatedData["name"] = field["value"]
                if key == "label" and field["label"] == "Last Name":
                    formatedData["name"] = formatedData["name"] + " " +field["value"]
                if key == "type" and field["type"] == "INPUT_EMAIL":
                    formatedData["email"] = field["value"]
                if key == "type" and field[key] == "INPUT_PHONE_NUMBER":
                    formatedData["phone"] = field["value"]
                if key == "label" and field["label"] == "Current Role":
                    formatedData["currentJob"] = field["value"]
                if key == "label" and field[key] == "Target Role":
                    formatedData["targetJob"] = field["value"]
                if key == "label" and field[key] == "List skills":
                    formatedData["skills"] = field["value"]
                if key == "key" and field[key] == "question_rDdQeM":
                    formatedData["experience"] = field["value"]
                if key == "key" and field[key] == "question_4aoYMb":
                    formatedData["education"] = []
                    for option in field["options"]:
                        if option["id"] in field["value"]:
                            formatedData["education"].append(option["text"])



                if key != "options":
                    print(key + ": ", end="")
                    print(field[key])
                else:
                    print(key + ": " , end= "")
                    print(field[key][0])
            print("}")
        formatedData["skills"] = formatedData["skills"].split(sep=',')


        if User.objects.get(id=0) == None:
            User.objects.create(name=formatedData["name"], email=formatedData["email"], phone_number=formatedData["phone"],
                                job_title=formatedData["currentJob"], targetJob=formatedData["targetJob"], skills=','.join(formatedData["skills"]),
                                experience=formatedData["experience"], education=','.join(formatedData["education"]))

        return Response("Success")



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

