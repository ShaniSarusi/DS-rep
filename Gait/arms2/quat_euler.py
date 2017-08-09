import math as m





quat = []
quat1 = ([-0.6766063216691801, -0.20615885929798813, -0.6825464678578937, 0.18393675387199554])
quat.append(quat1)
# -0.6768486705847616	-0.20597253287976414	-0.6824071228359866	0.1837708125068982
# -0.677135472024189	-0.2057253476683793	-0.6822454422394272	0.18359136798017311
# -0.6774441037975982	-0.20544485892415318	-0.6820743222101625	0.1834026039974656
# -0.677756879433619	-0.2051382642827877	-0.6819102624065759	0.18320016084065743

euler = []
euler1 = ([-0.04240546710789204, -0.04689168967306614, -0.022751548700034617])
euler.append(euler1)
# -0.06024353578686714	-0.06334116719663142	-0.023499252274632454
# -0.07444990612566471	-0.0766930118203163	-0.02830591667443514
# -0.08256782591342926	-0.08278145492076874	-0.030762656591832638
# -0.09004486277699471	-0.0823541983962059	-0.03289895225316286


def e_to_q(pitch, roll, yaw):
    #t0 = m.degrees(m.cos(yaw * 0.5))
    t0 = m.cos(m.degrees(yaw * 0.5))
    t1 = m.sin(m.degrees(yaw * 0.5))
    t2 = m.cos(m.degrees(roll * 0.5))
    t3 = m.sin(m.degrees(roll * 0.5))
    t4 = m.cos(m.degrees(pitch * 0.5))
    t5 = m.sin(m.degrees(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5
    return [w, x, y, z]


def Quaternion_toEulerianAngle(x, y, z, w):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = m.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = m.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = m.atan2(t3, t4)

    return X, Y, Z

# Code
q2 = e_to_q(euler1[0], euler1[1], euler1[2])
print(q2)
print(quat1)

e = Quaternion_toEulerianAngle(quat1[1], quat1[2], quat1[3], quat1[0])
print(e)
print(euler1)

	# t0 = std::cos(yaw * 0.5)
	# t1 = std::sin(yaw * 0.5)
	# t2 = std::cos(roll * 0.5)
	# t3 = std::sin(roll * 0.5)
	# t4 = std::cos(pitch * 0.5)
	# t5 = std::sin(pitch * 0.5)