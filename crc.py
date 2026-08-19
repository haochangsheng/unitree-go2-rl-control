import struct

from unitree_go.msg import LowCmd

class CRC():
    def __init__(self):
        #4 bytes aligned, little-endian format.
        #size 812
        self.__packFmtLowCmd = '<4B4IH2x' + 'B3x5f3I' * 20 + '4B' + '55Bx2I'
    
    def Crc(self, msg):
        return self.__Crc32(self.__PackLowCmd(msg))

    def __PackLowCmd(self, cmd: LowCmd):
        origData = []
        origData.extend(cmd.head)
        origData.append(cmd.level_flag)
        origData.append(cmd.frame_reserve)
        origData.extend(cmd.sn)
        origData.extend(cmd.version)
        origData.append(cmd.bandwidth)

        for i in range(20):
            origData.append(cmd.motor_cmd[i].mode)
            origData.append(cmd.motor_cmd[i].q)
            origData.append(cmd.motor_cmd[i].dq)
            origData.append(cmd.motor_cmd[i].tau)
            origData.append(cmd.motor_cmd[i].kp)
            origData.append(cmd.motor_cmd[i].kd)
            origData.extend(cmd.motor_cmd[i].reserve)

        origData.append(cmd.bms_cmd.off)
        origData.extend(cmd.bms_cmd.reserve)

        origData.extend(cmd.wireless_remote)
        origData.extend(cmd.led)
        origData.extend(cmd.fan)
        origData.append(cmd.gpio)
        origData.append(cmd.reserve)
        origData.append(cmd.crc)

        return self.__Trans(struct.pack(self.__packFmtLowCmd, *origData))

    def __Trans(self, packData):
        calcData = []
        calcLen = ((len(packData)>>2)-1)

        for i in range(calcLen):
            d = ((packData[i*4+3] << 24) | (packData[i*4+2] << 16) | (packData[i*4+1] << 8) | (packData[i*4]))
            calcData.append(d)

        return calcData

    def _crc_py(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04c11db7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for b in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1
        
        return crc

    def __Crc32(self, data):
        return self._crc_py(data)