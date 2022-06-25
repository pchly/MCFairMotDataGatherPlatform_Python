import math

class GyroBias():
    def __init__(self):
        self.calibrationUpdates = 0

        self.minX = 1000
        self.minY = 1000
        self.minZ = 1000
        self.maxX = -1000
        self.maxY = -1000
        self.maxZ = -1000

        self.x = 0
        self.y = 0
        self.z = 0

        self.isSet = False

    # def reset(self):
    def update(self,gx,gy,gz):

        if self.calibrationUpdates < 800:

            self.maxX = max(gx, self.maxX)
            self.maxY = max(gy, self.maxY)
            self.maxZ = max(gz, self.maxZ)

            self.minX = min(gx, self.minX)
            self.minY = min(gy, self.minY)
            self.minZ = min(gz, self.minZ)

            self.calibrationUpdates = self.calibrationUpdates + 1
            # return False
        elif self.calibrationUpdates == 800:
            self.x = (self.maxX + self.minX) / 2.0
            self.y = (self.maxY + self.minY) / 2.0
            self.z = (self.maxZ + self.minZ) / 2.0
            self.calibrationUpdates = self.calibrationUpdates + 1

            # / *std::cout << "BIAS-X: " << minX << " - " << maxX << std::endl;
            # std::cout << "BIAS-Y: " << minY << " - " << maxY << std::endl;
            # std::cout << "BIAS-Z: " << minZ << " - " << maxZ << std::endl;
            # std::cout << "BIAS: " << x << " " << y << " " << z << std::endl;
            # * /
            self.isSet = True

            # return True
        # else:
        #     return False


class rotation_estimator():

    def __init__(self):
        # theta is the angle of camera rotation in x, y and z components
        self.theta = [0.0,0.0,0.0]
        self.theta_from_gyro = [0.0,0.0,0.0]
        self.theta_from_accel = [0.0, 0.0, 0.0]
        #/ *alpha indicates the part that gyro and accelerometer take in computation of theta;
        #higher alpha gives more weight to gyro, but too high values cause drift; lower alpha gives
        # more weight to accelerometer, which is more sensitive to disturbances * /
        self.Kp = 100  # 比例增益控制加速度计/磁强计的收敛速度
        self.Ki = 0.002  # 积分增益控制陀螺偏差的收敛速度
        self.halfT = 0.0025  # 采样周期的一半

        # 传感器框架相对于辅助框架的四元数(初始化四元数的值)
        self.q0 = 1
        self.q1 = 0
        self.q2 = 0
        self.q3 = 0

        # 由Ki缩放的积分误差项(初始化)
        self.exInt = 0
        self.eyInt = 0
        self.ezInt = 0

        self.alpha = 0.98
        self.firstGyro = True
        self.firstAccel = True
        #// Keeps the arrival time of previous gyro frame
        self.last_ts_gyro = 0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.bias = GyroBias()






    #Function to calculate the change in angle of motion based on data from gyro
    def process_gyro(self,gyro_data, ts):

        #On the first iteration, use only data from accelerometer to set the camera's initial position
        if self.firstGyro:
            self.firstGyro = False
            self.last_ts_gyro = ts
        else:
            #Holds the change in angle, as calculated from gyro
            gyro_angle = [0.0,0.0,0.0]
            # Initialize gyro_angle with data from gyro
            gyro_angle[0] = round(gyro_data.x,2) # Pitch
            gyro_angle[1] = round(gyro_data.y,2) # Yaw
            gyro_angle[2] = round(gyro_data.z,2) # Roll

            #Compute the difference between arrival times of previous and current gyro frames
            dt_gyro = (ts - self.last_ts_gyro) / 1000.0
            self.last_ts_gyro = ts
            print("dt_gyro:", dt_gyro)
            #Change in angle equals gyro measures * time passed since last measurement
            # gyro_angle = gyro_angle * float(dt_gyro)
            gyro_angle[0] = gyro_angle[0] * float(dt_gyro)
            gyro_angle[1] = gyro_angle[1] * float(dt_gyro)
            gyro_angle[2] = gyro_angle[2] * float(dt_gyro)

            #Apply the calculated change of angle to the current angle (theta)
            # self.theta.add(-gyro_angle.z, -gyro_angle.y, gyro_angle.x)
            self.theta_from_gyro[0] = self.theta_from_gyro[0] - gyro_angle[0]
            self.theta_from_gyro[1] = self.theta_from_gyro[1] - gyro_angle[1]
            self.theta_from_gyro[2] = self.theta_from_gyro[2] + gyro_angle[2]
            self.theta[0] = self.theta_from_gyro[0]
            self.theta[1] = self.theta_from_gyro[1]
            self.theta[2] = self.theta_from_gyro[2]


    def process_accel(self,accel_data):
        a_x = round(accel_data.x, 2)
        a_y = round(accel_data.y, 2)
        a_z = round(accel_data.z,2)

        #Holds the angle as calculated from accelerometer data
        accel_angle = [0.0,0.0,0.0]
        #Calculate rotation angle from accelerometer data
        accel_angle[2] = math.atan2(a_y, a_z)
        accel_angle[0] = math.atan2(a_x, math.sqrt(a_y * a_y + a_z * a_z))
        #If it is the first iteration, set initial pose of camera according to
        # accelerometer data (note the different handling for Y axis)
        if self.firstAccel:
            self.firstAccel = False
            self.theta = accel_angle
            #Since we can't infer the angle around Y axis using
            # accelerometer data, we'll use PI as a convetion for the initial pose
            self.theta[1] = math.pi
        else:
            self.theta[0] = self.theta_from_gyro[0] * self.alpha - accel_angle[0] * (1 - self.alpha)
            self.theta[1] = self.theta_from_gyro[1] * self.alpha - accel_angle[1] * (1 - self.alpha)
            self.theta[2] = self.theta_from_gyro[2] * self.alpha + accel_angle[2] * (1 - self.alpha)
        #     # Apply Complementary Filter:
        #     #- high-pass filter = theta * alpha:  allows short-duration signals to pass through while filtering out signals
        #      # that are steady over time, is used to cancel out drift.
        #     #- low-pass filter = accel * (1- alpha): lets through long term changes, filtering out short term fluctuations
        #     self.theta[0] = self.theta[0] * self.alpha + accel_angle[0] * (1 - self.alpha)
        #     self.theta[2] = self.theta[2] * self.alpha + accel_angle[2] * (1 - self.alpha)


    def get_theta(self):

        # self.theta[0] = self.theta[0] * 180 / math.pi
        # self.theta[1] = self.theta[1] * 180 / math.pi
        # self.theta[2] = self.theta[2] * 180 / math.pi
        return self.theta

    def Update_IMU(self,ax,ay,az,gx,gy,gz):
        # print(q0)

        #测量正常化
        norm = math.sqrt(ax*ax+ay*ay+az*az)
        #单元化
        ax = ax/norm
        ay = ay/norm
        az = az/norm

        #估计方向的重力
        vx = 2*(self.q1*self.q3 - self.q0*self.q2)
        vy = 2*(self.q0*self.q1 + self.q2*self.q3)
        vz = self.q0*self.q0 - self.q1*self.q1 - self.q2*self.q2 + self.q3*self.q3

        #错误的领域和方向传感器测量参考方向之间的交叉乘积的总和
        ex = (ay*vz - az*vy)
        ey = (az*vx - ax*vz)
        ez = (ax*vy - ay*vx)

        #积分误差比例积分增益
        self.exInt += ex*self.Ki
        self.eyInt += ey*self.Ki
        self.ezInt += ez*self.Ki

        #调整后的陀螺仪测量
        gx += self.Kp*ex + self.exInt
        gy += self.Kp*ey + self.eyInt
        gz += self.Kp*ez + self.ezInt

        #整合四元数
        self.q0 += (-self.q1*gx - self.q2*gy - self.q3*gz)*self.halfT
        self.q1 += (self.q0*gx + self.q2*gz - self.q3*gy)*self.halfT
        self.q2 += (self.q0*gy - self.q1*gz + self.q3*gx)*self.halfT
        self.q3 += (self.q0*gz + self.q1*gy - self.q2*gx)*self.halfT

        #正常化四元数
        norm = math.sqrt(self.q0*self.q0 + self.q1*self.q1 + self.q2*self.q2 + self.q3*self.q3)
        self.q0 /= norm
        self.q1 /= norm
        self.q2 /= norm
        self.q3 /= norm

        #获取欧拉角 pitch、roll、yaw
        pitch = math.asin(-2*self.q1*self.q3+2*self.q0*self.q2)*57.3
        roll = math.atan2(2*self.q2*self.q3+2*self.q0*self.q1,-2*self.q1*self.q1-2*self.q2*self.q2+1)*57.3
        # yaw = math.atan2(2*(self.q1*self.q2 + self.q0*self.q3),self.q0*self.q0+self.q1*self.q1-self.q2*self.q2-self.q3*self.q3)*57.3
        yaw = math.atan2(2 * (self.q1 * self.q2 + self.q0 * self.q3),-2*self.q2*self.q2-2*self.q3*self.q3+1) * 57.3

        self.theta[0] = pitch
        self.theta[1] = roll
        self.theta[2] = yaw

        # return pitch,roll,yaw

    def imutest(self,ts,gv,av):

        if self.firstGyro:
            self.firstGyro = False
            self.last_ts_gyro = ts
        else:
            # Compute the difference between arrival times of previous and current gyro frames
            dt_gyro = (ts - self.last_ts_gyro) / 1000.0
            self.last_ts_gyro = ts
            print("dt_gyro:", dt_gyro)

            self.bias.update(gv.x, gv.y, gv.z)

            ratePitch = gv.x - self.bias.x
            rateYaw = gv.y - self.bias.y
            rateRoll = gv.z - self.bias.z


            ratePitch = ratePitch * float(dt_gyro)
            rateYaw = rateYaw * float(dt_gyro)
            rateRoll = rateRoll * float(dt_gyro)

            self.roll += rateRoll
            self.pitch -= ratePitch
            self.yaw += rateYaw


        R = math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z)

        # accel_angle[2] = math.atan2(a_y, a_z)
        # accel_angle[0] = math.atan2(a_x, math.sqrt(a_y * a_y + a_z * a_z))
        newRoll = math.atan2(av.y, av.z)#math.acos(av.x / R)
        newYaw = math.acos(av.y / R)
        newPitch = math.atan2(av.x, math.sqrt(av.y * av.y + av.z * av.z))#math.acos(av.z / R)

        # if self.firstAccel:
        #     self.firstAccel = False
        #     self.roll = newRoll
        #     self.yaw = newYaw
        #     self.pitch = newPitch
        # else:
        #     self.roll = self.roll * 0.98 + newRoll * 0.02
        #     self.yaw = self.yaw * 0.98 + newYaw * 0.02
        #     self.pitch = self.pitch * 0.98 + newPitch * 0.02

        self.theta[0] = self.roll *57.3
        self.theta[1] = self.yaw *57.3
        self.theta[2] = self.pitch *57.3

