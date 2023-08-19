from __future__ import print_function

from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative, Command
from pymavlink import mavutil # Needed for command message definitions
import time
import math
import os
from sshkeyboard import listen_keyboard,stop_listening
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import blobconverter
from pipeline import *
import keyboard  
import threading
from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket
from depthai_sdk.classes import SpatialBbMappingPacket
from depthai_sdk.visualize.configs import BboxStyle, TextPosition
from depthai_sdk.visualize.objects import GenericObject
from depthai_sdk.visualize.visualizer import Visualizer
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
vehicle = connect("/dev/serial0", baud=921600, wait_ready=True)

global waypoint
waypoint=1
#0 Never change yaw
#1 Face next waypoint
#2 Face next waypoint except RTL
#3 Face along GPS course
vehicle.parameters['WP_YAW_BEHAVIOR'] = 1
#Precision Land enabled
#vehicle.parameters['PLND_ENABLED'] = 0
#vehicle.parameters['PLD_TYPE'] = 1
#vehicle.parameters['PLND_ALT_MAX'] = 7
track_ids=[] 
cmds = vehicle.commands
cmds.download()
cmds.wait_ready() # wait until download is complete.
print('total waypoints:', cmds.count)

homepos = vehicle.location.global_relative_frame
frame_center_global= None
object_center_global=None
frame_global=None

# init global vehicle_home
vehicle_home = None

# defining model index and class name mapping
first_search_class_names = ["Marsh-drop", "Marsh-pickup", "Med-drop", "Med-pickup", "Water-pickup", "Water-drop"]
class NameIndexMapping:
    model_index = 0
    model_name = ""
    model_seq = 1
    model_filename = ""

# model_index needs changed based on the model
nameIndexMappings = []
mapping1 = NameIndexMapping()
mapping1.model_index = 0
mapping1.model_name = "Marsh-drop"
mapping1.model_seq = 2
mapping1.model_filename = "marsh.waypoints"
nameIndexMappings.append(mapping1)

mapping2 = NameIndexMapping()
mapping2.model_index = 2
mapping2.model_name = "Marsh-pickup"
mapping2.model_seq = 1
mapping2.model_filename = "marsh.waypoints"
nameIndexMappings.append(mapping2)

mapping3 = NameIndexMapping()
mapping3.model_index = 3
mapping3.model_name = "Med-drop"
mapping3.model_seq = 2
mapping3.model_filename = "med.waypoints"
nameIndexMappings.append(mapping3)

mapping4 = NameIndexMapping()
mapping4.model_index = 5
mapping4.model_name = "Med-pickup"
mapping4.model_seq = 1
mapping4.model_filename = "med.waypoints"
nameIndexMappings.append(mapping4)

mapping5 = NameIndexMapping()
mapping5.model_index = 7
mapping5.model_name = "Water-drop"
mapping5.model_seq = 2
mapping5.model_filename = "water.waypoints"
nameIndexMappings.append(mapping5)

mapping6 = NameIndexMapping()
mapping6.model_index = 9
mapping6.model_name = "Water-pickup"
mapping6.model_seq = 1
mapping6.model_filename = "water.waypoints"
nameIndexMappings.append(mapping6)

# define class for model gps location
class ModelGpsLocation:
    model_name = ""
    model_filename = ""
    model_index = 0
    model_seq = 0
    model_altitude = 0.0
    model_latitude = 0.0
    model_longitude = 0.0

# init modelgps location list
gpslocations = []

# Starting calculating GPS location function
image_width = 1080  # Width of the captured image in pixels
image_height = 720  # Height of the captured image in pixels
camera_fov = 20.0   # 62.2  # Camera's horizontal field of view in degrees

# function to convert (X-coordinate, Y-coordinate) of the object in the image
# to global GPS location
def convertLocalPositionToGPSLocation(vehicle, object_x, object_y):
    # Get the current altitude
    altitude = vehicle.location.global_relative_frame.alt
    latitude = vehicle.location.global_relative_frame.lat
    longitude = vehicle.location.global_frame.lon

    drone_location = (latitude, longitude)  # GPS coordinates of the drone's location

    # Calculate the angle between the center of the image and the object
    angle_x = object_x * (camera_fov / image_width)
    angle_y = object_y * (camera_fov / image_height)

    # Calculate the distance to the object using trigonometry
    distance = altitude / math.tan(math.radians(angle_y))

    # Calculate the offset in latitude and longitude using the distance and angle
    offset_x = distance * math.sin(math.radians(angle_x))
    offset_y = distance * math.cos(math.radians(angle_x))

    # Calculate the GPS coordinates of the object
    # object_location = (drone_location[0] + offset_y, drone_location[1] + offset_x)
    object_location = (drone_location[0], drone_location[1])

    # return the GPS Global location
    dest = LocationGlobalRelative(object_location[0], object_location[1], altitude)
    # Print the values
    print("GPS Location's Latitude:", dest.lat)
    print("GPS Location's Longitude:", dest.lon)
    print("GPS Location's Altitude:", dest.alt)
    return dest

def findModelMapping(label):
    global nameIndexMappings
    obj = None
    for one in nameIndexMappings:
        if one.model_index == label:
            obj = one
            break
    return obj

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def calculate_frame_center(frame):
    height, width = frame.shape[:2]
    print("height:",height, "width:", width)
    center_x = width // 2
    center_y = height // 2
    return (center_x, center_y)

def calculate_center(xmin, ymin, xmax, ymax):
    if xmin is None or ymin is None  or xmax is None or ymax is None:
        return None 
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)
    return (center_x, center_y)

def calculate_difference(frame_center, object_center):
    diff_x = frame_center[0] - object_center[0]
    diff_y = frame_center[1] - object_center[1]
    return (diff_x, diff_y)


# needed home location from beginning
# home = vehicle.home_location
def save_mission_firstline(home):
    """
    Save a mission in the Waypoint file format
    """
    print("\nSave mission from Vehicle to file: %s" + str(home.lat) + str(home.lon) + str(home.alt))
    # print("home value " + str(home.lat))
    # Add file-format information
    output = 'QGC WPL 110\n'
    # Add home location as 0th waypoint
    output += "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (0, 1, 0,
                                                                    16, 0, 0, 0, 0, home.lat, home.lon, home.alt, 1)
    # output = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
    # 0, 1, 0, 16, 0, 0, 0, 0, 0, 0, 0, 1)
    return output

## saving gps info into a file
# 1  0   0   22  0.000000    0.000000    0.000000    0.000000    -35.361988  149.163753  00.000000  1
def save_mission_gpsinfo(seq, lat, lon, alt):
    commandline = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (str(seq),
            str(0), str(3), str(16), str(0.000000), str(0.000000), str(0.000000), str(0.000000),
            str(lat), str(lon), str(alt), str(1))
    return commandline

# trying to find another location obj
def findSecondLocation(loc, locs):
    for loc1 in locs:
        if loc.model_filename == loc1.model_filename and loc.model_seq != loc1.model_seq:
            return loc1

    return None

def save_mission_file(vhome, location, locations):
    missionfile = open(location.model_filename, 'w')
    if len(locations) == 0:
        text("inserting ********")
        firstline = save_mission_firstline(vhome)
        secondline = save_mission_gpsinfo(location.model_seq, location.model_latitude,
                             location.model_longitude, location.model_altitude)
        missionlines = firstline + secondline
        missionfile.write(missionlines)
    else:
       text("handling multiple mapping...")
       secondloc = findSecondLocation(location, locations)
       if secondloc is None:
           text("saving only one entry...")
           firstline = save_mission_firstline(vhome)
           secondline = save_mission_gpsinfo(location.model_seq, location.model_latitude,
                                location.model_longitude, location.model_altitude)
           missionlines = firstline + secondline
           missionfile.write(missionlines)
       else:
           text("saving two entries.....")
           if location.model_seq < secondloc.model_seq:
               text("first condition in saving two entries.....")
               firstline = save_mission_firstline(vhome)
               secondline = save_mission_gpsinfo(location.model_seq, location.model_latitude,
                                    location.model_longitude, location.model_altitude)
               thirdline = save_mission_gpsinfo(secondloc.model_seq, secondloc.model_latitude,
                                    secondloc.model_longitude, secondloc.model_altitude)
               missionlines = firstline + secondline + thirdline
               missionfile.write(missionlines)
           else:
               text("second condition in saving two entries.....")
               firstline = save_mission_firstline(vhome)
               secondline = save_mission_gpsinfo(secondloc.model_seq, secondloc.model_latitude,
                                    secondloc.model_longitude, secondloc.model_altitude)
               thirdline = save_mission_gpsinfo(location.model_seq, location.model_latitude,
                                    location.model_longitude, location.model_altitude)
               missionlines = firstline + secondline + thirdline
               missionfile.write(missionlines)
    missionfile.close()

def cb(packet: DetectionPacket):
    print("Callback function called")
    
    global frame_global, frame_center_global, object_center_global
    # access gps location list
    global gpslocations
    global vehicle_home
    frame_global = packet.frame
    frame_center_global = calculate_frame_center(packet.frame)
    
    # Draw red dot at frame center, drone loction
    cv2.circle(packet.frame, frame_center_global, radius=6, color=(0, 0, 255), thickness=-1)
    cross_length = 20  # adjust to the size you want

    # Draw green cross at frame center
    cv2.line(packet.frame, (frame_center_global[0] - cross_length, frame_center_global[1]), (frame_center_global[0] + cross_length, frame_center_global[1]), (0, 0, 255), thickness=3)
    cv2.line(packet.frame, (frame_center_global[0], frame_center_global[1] - cross_length), (frame_center_global[0], frame_center_global[1] + cross_length), (0, 0, 255), thickness=3)

   
    for det in packet.img_detections.detections:
        print("det call")
        bbox_norm = [det.xmin, det.ymin, det.xmax, det.ymax]
        xmin, ymin, xmax, ymax = frameNorm(packet.frame, bbox_norm)
        print("xmin:" , xmin, "ymin:", ymin, "xmax:" ,xmax, "ymax:", ymax)
        object_center_global = calculate_center(xmin, ymin, xmax, ymax)

        #Draw blue dot at object center only if a center was found
        if object_center_global is not None:
            print("draw frame center")
            cv2.circle(packet.frame, object_center_global, radius=10, color=(255, 0, 0), thickness=-1)
            # Draw bounding box for the detected object
            cv2.rectangle(packet.frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255) # white color
            font_thickness = 2
            label_name = str(det.label)
            # Put label name at the top right corner of the bounding box
            cv2.putText(packet.frame, label_name, (xmax, ymin), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.imshow(packet.name, packet.frame)

            # convert the center point to GPS
            gps_location = convertLocalPositionToGPSLocation(vehicle,
                                int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            locat = ModelGpsLocation()
            locat.model_index = det.label
            locat.model_longitude = gps_location.lon
            locat.model_latitude = gps_location.lat
            locat.model_altitude = gps_location.alt

            locmap = findModelMapping(det.label)
            if locmap is not None:
                # fillout all info
                locat.model_filename = locmap.model_filename
                locat.model_name = locmap.model_name
                locat.model_seq = locmap.model_seq
                save_mission_file(vehicle_home, locat, gpslocations)
                # if the original list is empty, adding the location
                if (len(gpslocations) == 0):
                   gpslocations.append(locat)
                 #  save_mission_file(vehicle_home, locat, gpslocations)
                else:
                   # if target list is not empty, only save GPS location which
                   # is not at list
                   found = False
                   for targetlocat in gpslocations:
                      if (det.label == targetlocat.model_index):
                         print("find same location: " + str(det.label))
                         found = True
                         break

                   if (found == False):
                      print("adding location ....")
                      # and adding model to list
                      gpslocations.append(locat)

                      # printing data into data_gpsinfo.txt
                      print(locat.model_name + ": " + str(locat.model_longitude) + ", "
                               + str(locat.model_latitude) + "\n")
                  #    save_mission_file(vehicle_home, locat, gpslocations)
            else:
                print("could not find mapping")

        difference = calculate_difference(frame_center_global, object_center_global)
        
      #  print(f"Object center: {object_center_global}, Difference from frame center: {difference}")

PointWaterPick=0
PointMedDrop=0
PointMedPick=0
PointMarshDrop=0
PointMarshPick=0
PointWaterDrop=0
def camera2():
# open a file to write:
# starting oak camera
    with OakCamera() as oak:
        color = oak.create_camera('color')
        model_config = {
            'source': 'roboflow', 
            'model':'uas-target-krmde/1',
            'key':'BV4Xc6TaNPNR7A4hP7iy' 
        }
        nn = oak.create_nn(model_config, color, tracker=True)


        visualizer = oak.visualize(nn, fps=True)
        visualizer.__init__(scale=1)
    
        visualizer.detections(
            thickness=5,
            bbox_style=BboxStyle.RECTANGLE,
            label_position=TextPosition.TOP_RIGHT
        )
    

        oak.callback(nn.out.main, callback=cb, enable_visualizer=True) 
        oak.start(blocking=True)
        oak.poll
        for i in range(50):      
         #   print(f"frame_global:{frame_global},frame_center_global:{frame_center_global},object_center_global:{object_center_global}")
     # This is your main code where you can access the coordinates and draw the line
            if frame_center_global is not None and object_center_global is not None:
        # Draw the line on the frame
                cv2.line(frame_global, (int(frame_center_global[0]), int(frame_center_global[1])), 
                    (int(object_center_global[0]), int(object_center_global[1])), 
                    (255, 0, 0), 2)

            cv2.imshow('Frame', frame_global)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  

def returnDroneGPS():
    return vehicle.location.global_frame
def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")
         
    return targetlocation;

def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def land_to_home():
    pass

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    # after drone armed, taking home location
    global vehicle_home
    vehicle_home = vehicle.home_location
    print(str(vehicle_home.lat) +  str(vehicle_home.lon) + str(vehicle_home.alt))
    # print("Arming motors " + vehicle_home)
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:      
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)

def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for 
    the target position. This allows it to be called with different position-setting commands. 
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """
    
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)
    
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance

    while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
        #print "DEBUG: mode: %s" % vehicle.mode.name
        remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print("Distance to target: ", remainingDistance)
        if remainingDistance<=targetDistance*0.01: #Just below target, in case of undershoot.
            print("Reached target")
            break;
        time.sleep(2)


def condition_yaw(heading, relative=False):
    """
    Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).

    This method sets an absolute heading by default, but you can set the `relative` parameter
    to `True` to set yaw relative to the current yaw heading.

    By default the yaw of the vehicle will follow the direction of travel. After setting 
    the yaw using this function there is no way to return to the default yaw "follow direction 
    of travel" behaviour (https://github.com/diydrones/ardupilot/issues/2427)

    For more information see: 
    http://copter.ardupilot.com/wiki/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_condition_yaw
    """
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    direc = 1 if heading > 0 else -1
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        abs(heading),    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        direc,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)

def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors and
    for the specified duration.

    This uses the SET_POSITION_TARGET_LOCAL_NED command with a type mask enabling only 
    velocity components 
    (http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_local_ned).
    
    Note that from AC3.3 the message should be re-sent every second (after about 3 seconds
    with no message the velocity will drop back to zero). In AC3.2.1 and earlier the specified
    velocity persists until it is canceled. The code below should work on either version 
    (sending the message multiple times does not cause problems).
    
    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 

    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


def send_global_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.

    This uses the SET_POSITION_TARGET_GLOBAL_INT command with type mask enabling only 
    velocity components 
    (http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_global_int).
    
    Note that from AC3.3 the message should be re-sent every second (after about 3 seconds
    with no message the velocity will drop back to zero). In AC3.2.1 and earlier the specified
    velocity persists until it is canceled. The code below should work on either version 
    (sending the message multiple times does not cause problems).
    
    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_global_int_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, # lat_int - X Position in WGS84 frame in 1e7 * meters
        0, # lon_int - Y Position in WGS84 frame in 1e7 * meters
        0, # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
        # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
        velocity_x, # X velocity in NED frame in m/s
        velocity_y, # Y velocity in NED frame in m/s
        velocity_z, # Z velocity in NED frame in m/s
        0, 0, 0, # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 

    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)    

def set_velocity_body(vx, vy, vz, duration=2):
    """ Remember: vz is positive downward!!!
    http://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html
    
    Bitmask to indicate which dimensions should be ignored by the vehicle 
    (a value of 0b0000000000000000 or 0b0000001000000000 indicates that 
    none of the setpoint dimensions should be ignored). Mapping: 
    bit 1: x,  bit 2: y,  bit 3: z, 
    bit 4: vx, bit 5: vy, bit 6: vz, 
    bit 7: ax, bit 8: ay, bit 9:
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111, #-- BITMASK -> Consider only the velocities
            0, 0, 0,        #-- POSITION
            vx, vy, vz,     #-- VELOCITY
            0, 0, 0,        #-- ACCELERATIONS
            0, 0)
    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

def set_position_body_offset(dx, dy, dz):
    """ Remember: vz is positive downward!!!
    http://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html
    
    Bitmask to indicate which dimensions should be ignored by the vehicle 
    (a value of 0b0000000000000000 or 0b0000001000000000 indicates that 
    none of the setpoint dimensions should be ignored). Mapping: 
    bit 1: x,  bit 2: y,  bit 3: z, 
    bit 4: vx, bit 5: vy, bit 6: vz, 
    bit 7: ax, bit 8: ay, bit 9:
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0111110111111000, #-- BITMASK -> Consider only the velocities
            dx, dy, dz,        #-- POSITION
            0, 0, 0,     #-- VELOCITY
            0, 0, 0,        #-- ACCELERATIONS
            0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def goto_waypoint(nextwaypoint):
    cmds = vehicle.commands
    if nextwaypoint <= 0 or nextwaypoint > cmds.count:
        print('nextwaypoint is not correct:',nextwaypoint)
    elif nextwaypoint <= cmds.count:
        missionitem=vehicle.commands[nextwaypoint-1] #commands are zero indexed
        lat = missionitem.x
        lon = missionitem.y
        alt = missionitem.z
        targetLocation = LocationGlobalRelative(lat, lon, alt)
        currentLocation = vehicle.location.global_relative_frame
        targetDistance = get_distance_metres(currentLocation, targetLocation)
        vehicle.groundspeed = 3
        vehicle.simple_goto(targetLocation)
        while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
            #print("DEBUG: mode: %s" % vehicle.mode.name)
            remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
            print("Distance to target: ", remainingDistance)
            if remainingDistance<=targetDistance*0.05: #Just below target, in case of undershoot.
                print("Reached target")
                break;
            time.sleep(2)
        return nextwaypoint+1
    else:
        print("cmds.count is not correct:",cmds.count)

def press(key):
    global waypoint
    print(f"'{key}' is pressed")
    value = key
    #print("DEBUG: mode: %s" % vehicle.mode.name)
    if value == "p":
        vehicle.mode=VehicleMode("GUIDED")
        Program_Override()
    if value == 't':
        arm_and_takeoff(5)
    if vehicle.mode.name=="GUIDED":
        if value == 'w':
            #set_velocity_body(2,0,0)
            set_position_body_offset(1,0,0)
        elif value == 's':
            #set_velocity_body(-2,0,0)
            set_position_body_offset(-1,0,0)
        elif value == 'a':
            condition_yaw(-30,relative=True)
        elif value == 'd':
             condition_yaw(30,relative=True)
        elif value == 'l':
            vehicle.mode = VehicleMode("LAND")
            vehicle.close()
            stop_listening()
        elif value == 'r':
            vehicle.mode = VehicleMode("RTL")
            vehicle.close()
            stop_listening()
        elif value>="1" and value<="9":
            if vehicle.mode.name!="GUIDED":
                print('Not in GUIDED mode!')
            else:
                waypoint=goto_waypoint(int(value))
        elif value == "f":
            goto_target(10,2)
        elif value == "c":
            camera2()
        elif value== "m":
            Maunal_Override()
        elif value=="shift":
            send_ned_velocity(0,0,-0.5,1)
        elif value=="n":
            print("going to waypoint ",waypoint)
            text("flying to the next waypoint\n")
            waypoint=goto_waypoint(int(waypoint))
    else:
        print('please takeoff first!')
def press2(key):
    print(f"'{key}' is pressed")
    value = key
    #print("DEBUG: mode: %s" % vehicle.mode.name)
    if value == "p":
            Program_Override()
    if value == 't':
        arm_and_takeoff(5)
    if vehicle.mode.name=="GUIDED":
        if value == 'w':
            #set_velocity_body(2,0,0)
            set_position_body_offset(1,0,0)
        elif value == 's':
            #set_velocity_body(-2,0,0)
            set_position_body_offset(-1,0,0)
        elif value == 'a':
            condition_yaw(-30,relative=True)
        elif value == 'd':
             condition_yaw(30,relative=True)
        elif value == 'l':
            vehicle.mode = VehicleMode("LAND")
            vehicle.close()
            stop_listening()
        elif value == 'r':
            vehicle.mode = VehicleMode("RTL")
            vehicle.close()
            stop_listening()
        elif value>="1" and value<="9":
            if vehicle.mode.name!="GUIDED":
                print('Not in GUIDED mode!')
            else:
                goto_waypoint(int(value))
        
        elif value=="i":
            MotorIn()
        elif value == "c":
            camera2()
        elif value== "m":
            Maunal_Override()
        elif value=="shift":
            send_ned_velocity(0,0,-0.5,1)
        elif value=="f":
            text("Saving gps info...")


    elif value=="o":
            vehicle.mode.name=VehicleMode("GUIDED")
            MotorOut()     
    else:
        print('please takeoff first!')

import depthai as dai

# Clean up
def percision_land(x_meters,y_meters,z_meters):
    if z_meters>0:
        send_ned_velocity(x_meters,y_meters,0.1,0.5)
    else:
        vehicle.armed=False


def setWP(pts, alt):
    cmds = vehicle.commands
    cmds.add(
        Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0,
                0, 0, 0, pts.lat, pts.lon, alt))
def goto_target(x,y):
    cmds=vehicle.commands
    loc=get_location_metres(vehicle.location.global_frame,x,y,)
    vehicle.simple_goto(loc)
# Send the MAVLink command message
def tracking(x,y):
    #when camera is west, x is positive
    #when camera is east, x is negative
    #when camera is south, y is positive
    #when camera is north, y is negative
    print("")
    if x not in np.arange(0.05,0) and y not in np.arange(0.05,0):
        if x>0 and y>0: #when the camera is southwest relative to the drone
            send_ned_velocity(1,1,0,1)
        elif x>0 and y<0: #when the camera is northwest relative to the drone
            send_ned_velocity(1,-1,0,1)
        elif x<0 and y<0: #when the camera is northeast relative to the drone
            send_ned_velocity(-1,-1,0,1)
        elif x<0 and y>0:
            send_ned_velocity(-1,1,0,1)
def MotorOut():
    #define GPIO pins
    direction= 22 # Direction (DIR) GPIO Pin
    step = 23 # Step GPIO Pin
    EN_pin = 24 # enable pin (LOW to enable)

    # Declare a instance of class pass GPIO pins numbers and the motor type
    mymotortest = RpiMotorLib.A4988Nema(direction, step, (-1,-1,-1), "DRV8825")
    GPIO.setup(EN_pin,GPIO.OUT) # set enable pin as output

    ###########################
    # Actual motor control
    GPIO.output(EN_pin,GPIO.LOW) # pull enable to low to enable motor
    mymotortest.motor_go(True, # True= Mechanical box OUT- Clockwise, False= Mechanical box IN-Counter-Clockwise
                        "Full" , # Step type (Full,Half,1/4,1/8,1/16,1/32)
                        15000, # Full OUT/IN number of steps, take 8 second to complete full out/in
                        .0005, # step delay [sec]
                        False, # True = print verbose output 
                        .05) # initial delay [sec]

    GPIO.cleanup() # clear GPIO allocations after run 
def MotorIn():
    #define GPIO pins
    direction= 22 # Direction (DIR) GPIO Pin
    step = 23 # Step GPIO Pin
    EN_pin = 24 # enable pin (LOW to enable)

    # Declare a instance of class pass GPIO pins numbers and the motor type
    mymotortest = RpiMotorLib.A4988Nema(direction, step, (-1,-1,-1), "DRV8825")
    GPIO.setup(EN_pin,GPIO.OUT) # set enable pin as output

    ###########################
    # Actual motor control
    GPIO.output(EN_pin,GPIO.LOW) # pull enable to low to enable motor
    mymotortest.motor_go(False, # True= Mechanical box OUT- Clockwise, False= Mechanical box IN-Counter-Clockwise
                        "Full" , # Step type (Full,Half,1/4,1/8,1/16,1/32)
                        15000, # Full OUT/IN number of steps, take 8 second to complete full out/in
                        .0005, # step delay [sec]
                        False, # True = print verbose output 
                        .05) # initial delay [sec]

    GPIO.cleanup() # clear GPIO allocations after run 

def Maunal_Override():
    print("Manual Override!")
    vehicle.channels.overrides = {1: 1500, 2: 1500, 3: 1500, 4: 1500}
    vehicle.mode.name=VehicleMode("STABILIZE")

def Program_Override():
    print("Program Override!")
    vehicle.channels.overrides = {}
    vehicle.mode=VehicleMode("GUIDED")
def text(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.05 )

mission=int(input("Which mission would you like to run?"))
if mission==1:
    text("First Mission Starting...\n")
    time.sleep(0.5)
    print("input commad:")
    print("\t't' -- takeoff")
    print("\t'w' -- go ahead")
    print("\t's' -- go back")
    print("\t'a' -- turn left")
    print("\t'd' -- turn right")
    print("\t'1-9' --go to first 9 waypoints")
    print("\t'n' --next waypoint")
    print("\t'l' -- LAND and exit")
    print("\t'r' -- RTL and exit")
    print("\t'c' -- Camera(press x on the windows and then press q to exit)")
    print("\t'm' -- Manual Override")
    print("\t'p' -- Program Override")
    listen_keyboard(on_press=press,sequential=True)

    print('Stopped!')
elif mission==2:
    text("Second Mission Starting...\n")
    print("input command:")
    print("\t'w' -- go ahead")
    print("\t's' -- go back")
    print("\t'a' -- turn left")
    print("\t'd' -- turn right")
    print("\t'1-9' --go to first 9 waypoints")
    print("\t'n' --next waypoint")
    print("\t'o' -- StepMotorOut")
    print("\t'i' -- StepMotorIn")
    print("\t'l' -- LAND and exit")
    print("\t'r' -- RTL and exit")
    print("\t'c' -- Camera(press x on the windows and then press q to exit)")
    print("\t'm' -- Manual Override")
    print("\t'p' -- Program Override")
    listen_keyboard(on_press=press2,sequential=True)
