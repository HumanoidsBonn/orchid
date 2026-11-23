"""Event-driven perception images; failures never change action outcomes."""
from dataclasses import dataclass
import copy
import numpy as np
try:
    import cv2
except ImportError:  # Estimator width contracts do not depend on preview rendering.
    cv2 = None
import src.core.logger as log


@dataclass(frozen=True)
class ObservationImage:
    target: str
    image: np.ndarray
    camera_matrix: np.ndarray
    distortion: np.ndarray
    camera_from_world: np.ndarray
    world_frame: str
    header: object
    target_label: str | None = None


def capped_grasp_width(metadata):
    raw = float(metadata.get("raw_estimated_width_m",metadata.get("estimated_width_m",2.*metadata.get("estimated_radius_m",0.))))
    if not np.isfinite(raw) or raw < 0:
        raise ValueError("estimated grasp width must be finite and nonnegative")
    return {**metadata,"raw_estimated_width_m":raw,"estimated_width_m":min(.15,raw)}


def segmentation_image(image,bbox,mask,target):
    result=image.copy()
    overlay=result.copy()
    for polygon in mask or []:
        points=np.asarray(polygon,dtype=np.int32).reshape(-1,1,2)
        if len(points)>=3:
            cv2.fillPoly(overlay,[points],(60,210,90))
            cv2.polylines(result,[points],True,(60,255,100),2)
    result=cv2.addWeighted(overlay,.35,result,.65,0)
    ymin,xmin,ymax,xmax=map(int,bbox)
    cv2.rectangle(result,(xmin,ymin),(xmax,ymax),(60,255,100),2)
    cv2.putText(result,str(target).replace('_',' '),(max(0,xmin),max(18,ymin-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(60,255,100),2)
    return result


def publish_image(context,kind,image,header):
    from sensor_msgs.msg import CompressedImage
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    if context.ros2_node is None:
        return
    publisher=context.perception_preview_publishers.get(kind)
    if publisher is None:
        topic=f"/{context.robot_id.strip('/')}/visualization/{kind}/compressed"
        publisher=context.ros2_node.create_publisher(CompressedImage,topic,QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE))
        context.perception_preview_publishers[kind]=publisher
    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,70])
    if not ok:
        raise ValueError("preview JPEG encoding failed")
    message=CompressedImage(header=copy.deepcopy(header),format='jpeg',data=encoded.tobytes())
    publisher.publish(message)


def observation_preview(action,bbox,mask,world_frame):
    context=action._context
    target_label=getattr(action,'target_label',None)
    context.world_state.set('perception_preview_observation',None)
    try:
        publish_image(context,'segmentation',segmentation_image(action.processed_frame,bbox,mask,target_label or action.target),action.rgb_data.data.header)
    except Exception as exc:
        log.warn("Segmentation preview skipped: %s",exc)
    try:
        from scipy.spatial.transform import Rotation
        info=action.camera_info_data.data
        header=action.rgb_data.data.header
        camera_frame=header.frame_id or info.header.frame_id
        if not camera_frame or info.width<=0 or info.height<=0:
            raise ValueError('missing camera frame/calibration')
        projection=np.asarray(info.p,dtype=float).reshape(3,4)
        rectified=projection[0,0]>0 and projection[1,1]>0
        matrix=projection[:,:3].copy() if rectified else np.asarray(info.k,dtype=float).reshape(3,3).copy()
        if not np.isfinite(matrix).all() or matrix[0,0]<=0 or matrix[1,1]<=0:
            raise ValueError('invalid camera calibration')
        height,width=action.processed_frame.shape[:2]
        matrix[0,:]*=width/info.width
        matrix[1,:]*=height/info.height
        timestamp=header.stamp.sec+header.stamp.nanosec*1e-9
        if camera_frame==world_frame:
            transform=np.eye(4)
        else:
            if timestamp<=0:
                raise ValueError('missing observation timestamp')
            pose=context.sensor_data_center.get_pose_at(camera_frame,world_frame,timestamp=timestamp)
            if pose is None:
                raise ValueError('missing observation-time camera transform')
            transform=np.eye(4)
            transform[:3,3]=[pose['position'][k] for k in ('x','y','z')]
            transform[:3,:3]=Rotation.from_quat([pose['orientation'][k] for k in ('x','y','z','w')]).as_matrix()
        context.world_state.set('perception_preview_observation',ObservationImage(action.target,action.processed_frame.copy(),matrix,np.zeros(5) if rectified else np.asarray(info.d,dtype=float),np.linalg.inv(transform),world_frame,copy.deepcopy(header),target_label))
    except Exception as exc:
        log.warn("Grasp preview source unavailable: %s",exc)


def gripper_points(width):
    """Palm frame: opening along Y; fingers extend 9 cm along approach +Z."""
    width=min(.15,max(0.,float(width)))
    # Inner finger faces define opening; the crossbar lies at the palm origin.
    y=width/2
    return np.array([[0,-y-.01,0],[0,-y,0],[0,-y,.09],[0,-y-.01,.09],
                     [0,y,0],[0,y+.01,0],[0,y+.01,.09],[0,y,.09],
                     [0,0,-.06],[0,0,-.015]],dtype=float)


def grasp_image(source,palm_pose,width):
    points=gripper_points(width)
    camera_pose=source.camera_from_world@palm_pose
    camera_points=points@camera_pose[:3,:3].T+camera_pose[:3,3]
    if not np.isfinite(camera_points).all() or np.any(camera_points[:,2]<=.001):
        raise ValueError('grasp projects behind the camera')
    pixels,_=cv2.projectPoints(camera_points,np.zeros(3),np.zeros(3),source.camera_matrix,source.distortion)
    pixels=np.rint(pixels.reshape(-1,2)).astype(np.int32)
    if np.abs(pixels).max()>100000:
        raise ValueError('grasp projection outside supported image range')
    image=source.image.copy()
    for finger in (pixels[:4],pixels[4:8]):
        cv2.fillConvexPoly(image,finger,(255,190,30),cv2.LINE_AA)
    cv2.line(image,tuple(pixels[0]),tuple(pixels[5]),(255,190,30),3,cv2.LINE_AA)
    cv2.arrowedLine(image,tuple(pixels[8]),tuple(pixels[9]),(30,220,255),3,cv2.LINE_AA,tipLength=.3)
    caption=source.target_label or source.target.replace('_',' ')
    cv2.putText(image,f"{caption} | opening {min(.15,width)*100:.1f} cm",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,190,30),2)
    return image


def publish_grasp_preview(context,target,pose,frame,axis,width):
    try:
        from .hsr_pickup.planning import import_simple_grasp
        source=context.world_state.get('perception_preview_observation')
        if source is None or source.target!=target or source.world_frame!=frame:
            raise ValueError('no matching calibrated observation image')
        palm,_=import_simple_grasp(pose,axis)
        publish_image(context,'grasp',grasp_image(source,palm,width),source.header)
    except Exception as exc:
        log.warn("Grasp preview skipped: %s",exc)
