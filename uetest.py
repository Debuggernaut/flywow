import requests

UE = "http://127.0.0.1:30010"
ANIM = (
    "/Game/BigCompanyArchViz/Maps/UEDPIE_0_Maps_BigCompany"
    ".Maps_BigCompany:PersistentLevel.SkeletalMeshActor_1"
    ".SkeletalMeshComponent0.FlyAnimBP_C_0"
)
HEADERS = {"Content-Type": "application/json"}


def put(route, body):
    r = requests.put(f"{UE}{route}", json=body, headers=HEADERS, timeout=8)
    print(r.status_code, r.text[:600])
    r.raise_for_status()
    return r.json() if r.text else {}


print("describe")
put("/remote/object/describe", {"objectPath": ANIM})

print("read all")
put("/remote/object/property", {
    "objectPath": ANIM,
    "access": "READ_ACCESS",
})
print("read LF")
put("/remote/object/property", {
    "objectPath": ANIM,
    "propertyName": "LF",
    "access": "READ_ACCESS",
})

# 1) full struct + transaction
put("/remote/object/property", {
    "objectPath": ANIM,
    "propertyName": "LF",
    "access": "WRITE_ACCESS",
    "generateTransaction": True,
    "propertyValue" : {
        "LF": {"Pitch": 0, "Yaw" : 15.0, "Roll":0}
    }
})

# requests.put(
#     "http://127.0.0.1:30010/remote/preset/FlyLegs/property/LF",
#     json={"PropertyValue": {"Pitch": 45.0, "Yaw": 0.0, "Roll": 0.0}},
#     timeout=8,
# )
print("read LF")
put("/remote/object/property", {
    "objectPath": ANIM,
    "propertyName": "LF",
    "access": "READ_ACCESS",
})

#print("LF after", r.text)