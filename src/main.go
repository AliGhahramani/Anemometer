package main

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
	"github.com/immesys/bw2bind"
	l7g "github.com/immesys/chirp-l7g"
	"github.com/immesys/hamilton-decoder/common"
	"github.com/immesys/hcr"
	"github.com/immesys/ragent/ragentlib"
)

const serverVK = "MT3dKUYB8cnIfsbnPrrgy8Cb_8whVKM-Gtg2qd79Xco="
const ourEntity = ("\x32\x49\x4E\x95\x84\xDB\xE3\xE6\x10\x6D\xAB\x7F\xAB\x74\x04\x96" +
	"\x6F\x33\x2E\x13\xDF\x1F\xFE\xB1\x12\xF5\x1E\xCC\x90\x2A\x3C\x43" +
	"\xFA\x81\x48\x08\xDB\xC0\x57\xB0\xA1\x59\xB5\xEC\x4C\x68\x0A\x73" +
	"\xDD\xB6\xCD\x3E\x8D\x5A\xCE\xDE\xB4\x1C\x7A\xC5\x9E\x91\xB6\x6E" +
	"\xF2\x02\x08\x7B\xA5\xFF\xBD\xF8\x9E\xB2\x14\x03\x08\xDA\xA3\x2E" +
	"\xA2\x18\xD0\xE2\x16\x06\x0B\x62\x61\x6B\x65\x64\x65\x6E\x74\x69" +
	"\x74\x79\x00\x0A\xFE\xC6\x5D\x86\x89\x97\xFF\x6F\xCC\x34\xA9\xD4" +
	"\xFA\xD2\x67\x8A\x97\x5B\x38\x8B\xE7\x33\x1E\x58\x4E\x5F\x25\x41" +
	"\x48\xC1\xAE\x65\x73\x04\x43\xC2\x05\xC7\xD0\x4F\x1E\xE8\x47\x50" +
	"\xC5\x81\x8C\x97\x06\xDD\x08\x27\x00\x59\x20\xAC\xB3\x48\x88\x9F" +
	"\x9D\x01\x03")

func main() {
	infoc := color.New(color.FgBlue, color.Bold)
	errc := color.New(color.FgRed, color.Bold)
	infoc.Printf("anemomteer 2.8\n")
	borderIP := "52.9.16.254:28590"
	borderIPbyte, err := ioutil.ReadFile("./borderIP.txt")
	if err != nil {
		errc.Printf("Couldn't read borderIP text file, using default: %s\n", borderIP)
	} else {
		borderIP = strings.TrimSpace(fmt.Sprintf("%s", borderIPbyte))
		infoc.Printf("Using borderIP: %s\n", borderIP)
	}
	go func() {
		defer func() {
			r := recover()
			if r != nil {
				errc.Printf("failed to connect ragent: %v", r)
				os.Exit(1)
			}
		}()
		ragentlib.DoClientER([]byte(ourEntity), "52.9.16.254:28590", serverVK, "127.0.0.1:28588")
	}()
	time.Sleep(200 * time.Millisecond)
	go dohamz()
	time.Sleep(1 * time.Second)
	lastNotify = time.Now()
	err = l7g.RunDPA([]byte(ourEntity)[1:], Initialize, OnNewData, Vendor, Version)
	errc.Printf("dpa error: %v\n", err)
}

func dohamz() {
	silentretries := 5
	var cl *bw2bind.BW2Client
	for {
		thiscl, err := bw2bind.Connect("127.0.0.1:28588")
		if err == nil {
			cl = thiscl
			break
		}
		silentretries--
		time.Sleep(200 * time.Millisecond)
		if silentretries == 0 {
			fmt.Printf("Could not connect: bad internet?\n")
			os.Exit(1)
		}
	}

	_, err := cl.SetEntity([]byte(ourEntity)[1:])
	if err != nil {
		panic(err)
	}
	ch := cl.SubscribeOrExit(&bw2bind.SubscribeParams{
		URI:       "ucberkeley/sasc/+/s.hamilton/+/i.l7g/signal/dedup",
		AutoChain: true,
	})
	for m := range ch {
		po := m.GetOnePODF("2.0.10.1")
		if po == nil {
			fmt.Printf("po mismatch\n")
			continue
		}
		im := common.Message{}
		po.(bw2bind.MsgPackPayloadObject).ValueInto(&im)
		if len(im.Payload) < 2 {
			continue
		}
		mtype := binary.LittleEndian.Uint16(im.Payload)
		if mtype == 8 {
			Handle(m, &im)
		}
	}
}

func Handle(sm *bw2bind.SimpleMessage, im *common.Message) {
	if len(im.Payload) != 52 {
		fmt.Printf("dropping hamilton-3c-v2 packet due to length mismatch: expected 52 got %d\n", len(im.Payload))
	}
	serial := binary.LittleEndian.Uint16(im.Payload[2:])
	moteinfo, err := hcr.GetMoteInfo(context.Background(), int(serial), "ZXwTL92wSzuP_3zRBnhW_ouBUfTVHCYE")
	if err != nil {
		fmt.Printf("[%04x/%s] dropping hamilton-3c-v2 packet due to HCR error: %v\n", serial, "secret", err)
		return
	}
	block, err := aes.NewCipher(moteinfo.AESK[:16])
	if err != nil {
		fmt.Printf("[%04x/%s] dropping hamilton-3c-v2 packet due to error: %v\n", serial, "secret", err)
		return
	}
	iv := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F}
	dce := cipher.NewCBCDecrypter(block, iv)
	plaintext := make([]byte, len(im.Payload)-4)
	dce.CryptBlocks(plaintext, im.Payload[4:])
	for i := 40; i < 48; i++ {
		if plaintext[i] != 0 {
			fmt.Printf("[%04x/%s] dropping hamilton-3c-v2 packet because it looks like AES key is wrong\n", serial, "secrets")
			return
		}
	}

	f_uptime := binary.LittleEndian.Uint64(plaintext[0:8])
	f_flags := binary.LittleEndian.Uint16(plaintext[8:10])
	f_acc_x := int16(binary.LittleEndian.Uint16(plaintext[10:12]))
	f_acc_y := int16(binary.LittleEndian.Uint16(plaintext[12:14]))
	f_acc_z := int16(binary.LittleEndian.Uint16(plaintext[14:16]))
	f_mag_x := int16(binary.LittleEndian.Uint16(plaintext[16:18]))
	f_mag_y := int16(binary.LittleEndian.Uint16(plaintext[18:20]))
	f_mag_z := int16(binary.LittleEndian.Uint16(plaintext[20:22]))
	f_tmp_die := int16(binary.LittleEndian.Uint16(plaintext[22:24]))
	f_tmp_val := binary.LittleEndian.Uint16(plaintext[24:26])
	f_hdc_tmp := int16(binary.LittleEndian.Uint16(plaintext[26:28]))
	f_hdc_rh := binary.LittleEndian.Uint16(plaintext[28:30])
	f_light_lux := binary.LittleEndian.Uint16(plaintext[30:32])
	f_buttons := binary.LittleEndian.Uint16(plaintext[32:34])
	f_occup := binary.LittleEndian.Uint16(plaintext[34:36])
	dat := make(map[string]float64)
	dat["uptime"] = float64(f_uptime)
	if f_flags&(1<<0) != 0 {
		//accel
		dat["acc_x"] = float64(f_acc_x) * 0.244
		dat["acc_y"] = float64(f_acc_y) * 0.244
		dat["acc_z"] = float64(f_acc_z) * 0.244

	}
	if f_flags&(1<<1) != 0 {
		dat["mag_x"] = float64(f_mag_x) * 0.1
		dat["mag_y"] = float64(f_mag_y) * 0.1
		dat["mag_z"] = float64(f_mag_z) * 0.1
	}
	if f_flags&(1<<2) != 0 {
		//TMP
		dat["tp_die_temp"] = float64(int16(f_tmp_die)>>2) * 0.03125
		uv := float64(int16(f_tmp_val)) * 0.15625
		dat["tp_voltage"] = uv
	}

	if f_flags&(1<<3) != 0 {
		//HDC
		rh := float64(f_hdc_rh) / 100
		t := float64(f_hdc_tmp) / 100
		dat["air_temp"] = t
		dat["air_rh"] = rh
		expn := (17.67 * t) / (t + 243.5)
		dat["air_hum"] = (6.112 * math.Pow(math.E, expn) * rh * 2.1674) / (273.15 + t)
	}
	if f_flags&(1<<4) != 0 {
		//LUX
		dat["lux"] = math.Pow(10, float64(f_light_lux)/(65536.0/5.0))
	}
	if f_flags&(1<<5) != 0 {
		dat["button_events"] = float64(f_buttons)
	}
	if f_flags&(1<<6) != 0 {
		dat["presence"] = float64(f_occup) / 32768
	}

	dat["time"] = float64(im.Brtime)
	kz := []string{}
	for k, _ := range dat {
		kz = append(kz, k)
	}
	sort.StringSlice(kz).Sort()
	vz := []MSR{}
	for _, k := range kz {
		vz = append(vz, MSR{Mac: fmt.Sprintf("%04x", int(serial)),
			Pfx:  fmt.Sprintf("hamilton/%04x", int(serial)),
			Name: k,
			Time: time.Unix(0, im.Brtime),
			Val:  float64(dat[k]),
			Unit: "hu",
		})
	}
	LogMSR(vz)
}

var lastNotify time.Time

var mra = make(map[string]*room_anemometer)

const Vendor = "berkeley"
const Version = "ref_1_4"

type MSR struct {
	Mac  string
	Pfx  string
	Name string
	Time time.Time
	Val  float64
	Unit string
}

const SERVER = "http://lp.cal-sdb.org:8086/v1"

var glock = sync.Mutex{}

func LogMSR(vz []MSR) {
	glock.Lock()
	for _, z := range vz {
		ln := fmt.Sprintf("%s\t/anemometer/%s/%s\t%s=%f", z.Time, z.Mac, z.Pfx, z.Name, z.Val)
		fmt.Println(ln)
	}
	glock.Unlock()
	//
	// //then := time.Now()
	// ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	// defer cancel()
	// client := &http.Client{}
	//
	// req, err := http.NewRequest("POST", SERVER, &buf)
	// if err != nil {
	// 	panic(err)
	// }
	// req = req.WithContext(ctx)
	// resp, err := client.Do(req)
	// if err != nil {
	// 	fmt.Printf("Failed err=%s\n", err)
	// } else {
	// 	defer resp.Body.Close()
	// 	if resp.StatusCode != 200 {
	// 		bdy, _ := ioutil.ReadAll(resp.Body)
	// 		fmt.Printf("Failed: %s\n", string(bdy))
	// 	} else {
	// 		//	fmt.Printf("Published ok %s\n", time.Now().Sub(then))
	// 	}
	// }
}

type room_anemometer struct {
	//map the port number to the matrix index
	port_to_idx [4]int32
	//separation between parts in terms of index in microns
	s_matrix [4][4]float32
	//tofs in microseconds
	tof_matrix [4][4]float32
	//expected tofs
	tof_expected [4][4]float32
	//tof_expected IIR coefficient
	tof_e_coeff [4][4]float32
	//allowed range of tofs (can be +/- of this value)
	tof_range [4][4]float32
	//number of good tofs
	count_good [4][4]int32
	//component velocities in m/s
	vel_matrix [4][4]float32
	//scale factors from components to cardinal
	v_scales [3][4][4]float32
	//	vy_scales [4][4]float32
	//	vz_scales [4][4]float32
	//raw cardinal velocities m/s
	vxyz_raw [3]float32
	//stored offset values
	vxyz_offset [3]float32
	//calibrated velocities
	vxyz_cal [3]float32
	//filtered result, output to application
	vxyz_filt [3]float32
	//number of received samples
	num_samples int32
	trace_sum   [4]float32
	trace_filt  [4]float32
	trace_diff  [4]float32
	cal_state   int8
	//anemometer type, 1 = room, 2 = 6" duct
	mode int32
}

func NewDuctAnemometer() *room_anemometer {
	ra := room_anemometer{}
	ra.num_samples = 0
	ra.mode = 2

	//initialize port to index according to geometry
	ra.port_to_idx[0] = 2 //port 0 downstream top
	ra.port_to_idx[1] = 1 //port 1 is upstream bottom
	ra.port_to_idx[2] = 0 //port 2 is upstream top
	ra.port_to_idx[3] = 3 //port 3 is downstream bottom

	baserange := float32(152400.0) //in microns
	ra.s_matrix[0][1] = baserange
	ra.s_matrix[0][2] = baserange * float32(math.Sqrt(2))
	ra.s_matrix[0][3] = baserange * float32(math.Sqrt(3))
	ra.s_matrix[1][2] = baserange * float32(math.Sqrt(3))
	ra.s_matrix[1][3] = baserange * float32(math.Sqrt(2))
	ra.s_matrix[2][3] = baserange

	for i := 0; i < 4; i++ {
		ra.s_matrix[i][i] = 0.0
		ra.tof_matrix[i][i] = 1.0e-12

	}

	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			ra.s_matrix[j][i] = ra.s_matrix[i][j]
		}
		for k := 0; k < 4; k++ {
			ra.tof_e_coeff[i][k] = 0.5
			ra.count_good[i][k] = 1
			ra.tof_range[i][k] = float32(5.0 + 10.0/ra.count_good[i][k])
			ra.tof_matrix[i][k] = ra.s_matrix[i][k] / 343.0
			ra.tof_expected[i][k] = ra.tof_matrix[i][k]
			ra.tof_expected[i][k] = ra.tof_matrix[i][k]
		}
	}

	//0 is the top
	ra.v_scales[0][0][2] = float32(math.Cos(45.0 * math.Pi / 180.0))

	//1 is the bottom
	ra.v_scales[1][1][3] = float32(math.Cos(45.0 * math.Pi / 180.0))
	//2 is the diagonal
	ra.v_scales[2][0][3] = float32(math.Cos(45.0*math.Pi/180.0) * math.Cos(30.0*math.Pi/180.0))
	ra.v_scales[2][1][2] = float32(math.Cos(45.0*math.Pi/180.0) * math.Cos(30.0*math.Pi/180.0))

	//flip the matrix across the identity axis
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			for k := 0; k < 3; k++ {
				//possibly big bug here
				ra.v_scales[k][j][i] = -ra.v_scales[k][i][j]
			}
		}
	}
	return &ra
}

func NewRoomAnemometer() *room_anemometer {
	ra := room_anemometer{}
	ra.num_samples = 0
	ra.mode = 1

	//initialize port to index according to geometry
	ra.port_to_idx[0] = 1 //port 0 is B in doc
	ra.port_to_idx[1] = 3 //port 1 is D
	ra.port_to_idx[2] = 0 //A
	ra.port_to_idx[3] = 2 //C

	//initialize s_matrix, tof_matrix.
	//Other matrixes are already initialized to 0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			ra.tof_e_coeff[i][j] = 0.5
			ra.s_matrix[i][j] = 60000.0
			ra.count_good[i][j] = 1
			ra.tof_matrix[i][j] = ra.s_matrix[i][j] / 343.0
			ra.tof_expected[i][j] = ra.tof_matrix[i][j]
			ra.tof_range[i][j] = float32(5.0 + 10.0/ra.count_good[i][j])
			if i == j {
				ra.s_matrix[i][j] = 0.0
				ra.tof_matrix[i][j] = 1.0e-12 //prevent divide by zero
			}
		}
	}
	ra.v_scales[0][0][2] = float32(math.Cos(30.0 * math.Pi / 180.0))
	ra.v_scales[0][1][2] = float32(math.Cos(30.0 * math.Pi / 180.0))
	ra.v_scales[0][0][3] = float32(math.Cos(54.74*math.Pi/180.0) * math.Sin(60.0*math.Pi/180.0))
	ra.v_scales[0][1][3] = float32(math.Cos(54.74*math.Pi/180.0) * math.Sin(60.0*math.Pi/180.0))
	ra.v_scales[0][2][3] = float32(-math.Cos(54.74 * math.Pi / 180.0))

	ra.v_scales[1][0][1] = 1.0
	ra.v_scales[1][0][2] = float32(math.Sin(30.0 * math.Pi / 180.0))
	ra.v_scales[1][0][3] = float32(math.Cos(54.74*math.Pi/180.0) * math.Cos(60.0*math.Pi/180.0))
	ra.v_scales[1][1][2] = float32(-math.Sin(30.0 * math.Pi / 180.0))
	ra.v_scales[1][1][3] = float32(-math.Cos(54.74*math.Pi/180.0) * math.Cos(60.0*math.Pi/180.0))

	ra.v_scales[2][0][3] = float32(math.Sin(54.74 * math.Pi / 180.0))
	ra.v_scales[2][1][3] = float32(math.Sin(54.74 * math.Pi / 180.0))
	ra.v_scales[2][2][3] = float32(math.Sin(54.74 * math.Pi / 180.0))

	//flip the matrix across the identity axis
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			for k := 0; k < 3; k++ {
				//possibly big bug here
				ra.v_scales[k][j][i] = -ra.v_scales[k][i][j]
			}
		}
	}
	return &ra
}

func (ra *room_anemometer) setToFs(tof_us float32, txi int32, rxi int32, toprint bool) {
	if tof_us < ra.tof_expected[txi][rxi]+ra.tof_range[txi][rxi] && tof_us > ra.tof_expected[txi][rxi]-ra.tof_range[txi][rxi] {
		//ra.tof_matrix[txi][rxi] = tof_us
		ra.tof_e_coeff[txi][rxi] = 0.99 - 0.5/float32(ra.count_good[txi][rxi])
		ra.tof_expected[txi][rxi] = ra.tof_expected[txi][rxi]*ra.tof_e_coeff[txi][rxi] + tof_us*(1.0-ra.tof_e_coeff[txi][rxi])
		ra.tof_matrix[txi][rxi] = tof_us
		ra.tof_range[txi][rxi] = 5.0 + 10.0/float32(ra.count_good[txi][rxi])
		ra.count_good[txi][rxi]++
		if toprint {
			fmt.Printf("TOF expected[%d][%d]: %0.3f, range: %0.3f, coeff: %0.3f, good: %d\n", txi, rxi, ra.tof_expected[txi][rxi], ra.tof_range[txi][rxi], ra.tof_e_coeff[txi][rxi], ra.count_good[txi][rxi])
		}

	} else {
		if toprint {
			fmt.Printf("Bad sample, TOF expected[%d][%d]: %0.3f, range: %0.3f, coeff: %0.3f, good: %d\n", txi, rxi, ra.tof_expected[txi][rxi], ra.tof_range[txi][rxi], ra.tof_e_coeff[txi][rxi], ra.count_good[txi][rxi])
		}
	}

}

func (ra *room_anemometer) cardinalVelocities() {
	den := [3]float32{0.0, 0.0, 0.0}
	num := [3]float32{0.0, 0.0, 0.0}
	for k := 0; k < 3; k++ {
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				//weighted average, with weights equal to abs value of scale factor
				num[k] = num[k] + ra.vel_matrix[i][j]*ra.v_scales[k][i][j]*float32(math.Abs(float64(ra.v_scales[k][i][j])))
				den[k] = den[k] + float32(math.Abs(float64(ra.v_scales[k][i][j])))
			}
		}
		ra.vxyz_raw[k] = num[k] / den[k]
	}
}

func (ra *room_anemometer) filterTrace(coeff float32, i uint8) {
	if ra.num_samples <= 20 {
		ra.trace_filt[i] = ra.trace_sum[i]
	} else {
		ra.trace_filt[i] = ra.trace_filt[i]*coeff + ra.trace_sum[i]*(1-coeff)
	}
	ra.trace_diff[i] = ra.trace_sum[i] - ra.trace_filt[i]
}

func (ra *room_anemometer) filterVelocity(coeff float32) {
	for k := 0; k < 3; k++ {
		if ra.num_samples <= 20 {
			ra.vxyz_filt[k] = ra.vxyz_raw[k]
		} else {
			ra.vxyz_filt[k] = ra.vxyz_filt[k]*coeff + ra.vxyz_raw[k]*(1-coeff)
		}

	}
}

func (ra *room_anemometer) calibrateVelocity(samps int32) {

	for k := 0; k < 3; k++ {
		if ra.num_samples == samps {
			ra.vxyz_offset[k] = ra.vxyz_filt[k]
		}
		ra.vxyz_cal[k] = ra.vxyz_filt[k] - ra.vxyz_offset[k]
	}

}

func Initialize(emit l7g.Emitter) {
	//We actually do not do any initialization in this implementation, but if
	//you want to, you can do it here.
}

// OnNewData encapsulates the algorithm. You can store the emitter and
// use it asynchronously if required. You can see the documentation for the
// parameters at https://godoc.org/github.com/immesys/chirp-l7g
func OnNewData(popHdr *l7g.L7GHeader, h *l7g.ChirpHeader, emit l7g.Emitter) {
	vz := []MSR{}
	// Define some magic constants for the algorithm
	duct := false
	magic_count_tx := -4.8 //-3.125
	//	threshold := uint64(1400 * 1400)
	//	fmt.Printf(".\n")
	//Room anemometers have build numbers like 110, 120, 130
	//duct anemometers have build numbers like 115, 125, 135
	if h.Build%10 == 5 {
		//This is a duct anemometer, and this algorithm is not yet updated to
		//deal with that
		//		fmt.Printf("dropping duct anemometer data\n")
		//corect magic_count_tx to account for the 13 dropped samples in the duct trace
		magic_count_tx = magic_count_tx + 13.0
		duct = true
	}

	ra, ok := mra[popHdr.Srcmac]
	if ok == false {
		//fmt.Printf("No key for: %s, creating new RA\n", popHdr.Srcmac)
		if duct {
			mra[popHdr.Srcmac] = NewDuctAnemometer()
		} else {
			mra[popHdr.Srcmac] = NewRoomAnemometer()
		}

		ra = mra[popHdr.Srcmac]
		//fmt.Println(ra)
	}

	// Create our output data set. For this reference implementation,
	// we emit one TOF measurement for every raw TOF sample (no averaging)
	// so the timestamp is simply the raw timestamp obtained from the
	// Border Router. We also identify the sensor simply from the mac address
	// (this is fine for most cases)
	odata := l7g.OutputData{
		Timestamp: popHdr.Brtime,
		Sensor:    popHdr.Srcmac,
	}
	toprint := false
	isprimary := false
	poptime := time.Unix(0, popHdr.Brtime)
	SetOffset := 30 * time.Millisecond * time.Duration(h.Primary)
	// For each of the four measurements in the data set
	vz = append(vz, MSR{Mac: popHdr.Srcmac,
		Pfx:  fmt.Sprintf("calpulse"),
		Name: "calpulse",
		Time: poptime.Add(SetOffset),
		Val:  float64(h.CalPulse),
		Unit: "calpulse",
	})
	for set := 0; set < 4; set++ {
		isprimary = false
		// For now, ignore the data read from the ASIC in TXRX
		if int(h.Primary) == set {
			isprimary = true
		}

		// alias the data for readability. This is the 70 byte dataset
		// read from the ASIC
		data := h.Data[set]

		//The first six bytes of the data
		tof_sf := binary.LittleEndian.Uint16(data[0:2])
		tof_est := binary.LittleEndian.Uint16(data[2:4])
		intensity := binary.LittleEndian.Uint16(data[4:6])

		//CalPulse is in microseconds
		freq := float64(tof_sf) / 2048 * float64(h.CalRes[set]) / (float64(h.CalPulse) / 1000)
		vz = append(vz, MSR{Mac: popHdr.Srcmac,
			Pfx:  fmt.Sprintf("freq/%d_to_%d", h.Primary, set),
			Name: "freq",
			Time: poptime.Add(SetOffset),
			Val:  float64(freq),
			Unit: "freq",
		})
		vz = append(vz, MSR{Mac: popHdr.Srcmac,
			Pfx:  fmt.Sprintf("calres/%d_to_%d", h.Primary, set),
			Name: "calres",
			Time: poptime.Add(SetOffset),
			Val:  float64(h.CalRes[set]),
			Unit: "calres",
		})
		SampleOffset := int(100000000000 / freq)
		//Load the complex numbers
		iz := make([]int16, 16)
		qz := make([]int16, 16)
		for i := 0; i < 16; i++ {
			qz[i] = int16(binary.LittleEndian.Uint16(data[6+4*i:]))
			iz[i] = int16(binary.LittleEndian.Uint16(data[6+4*i+2:]))

			vz = append(vz, MSR{Mac: popHdr.Srcmac,
				Pfx:  fmt.Sprintf("raw/%d_to_%d", h.Primary, set),
				Name: "real",
				Time: poptime.Add(SetOffset).Add(time.Duration(SampleOffset * i)),
				Val:  float64(qz[i]),
				Unit: "real",
			})
			vz = append(vz, MSR{Mac: popHdr.Srcmac,
				Pfx:  fmt.Sprintf("raw/%d_to_%d", h.Primary, set),
				Name: "im",
				Time: poptime.Add(SetOffset).Add(time.Duration(SampleOffset * i)),
				Val:  float64(iz[i]),
				Unit: "imaginary",
			})
			arg := float64(qz[i])*float64(qz[i]) + float64(iz[i])*float64(iz[i])
			mag := math.Sqrt(arg)
			//fmt.Printf("qz=%v iz=%v arg=%v mag=%v\n", qz[i], iz[i], arg, mag)
			vz = append(vz, MSR{Mac: popHdr.Srcmac,
				Pfx:  fmt.Sprintf("raw/%d_to_%d", h.Primary, set),
				Name: "mag",
				Time: poptime.Add(SetOffset).Add(time.Duration(SampleOffset * i)),
				Val:  mag,
				Unit: "magnitude",
			})
			_ = mag

		}

		//Find the largest complex magnitude (as a square). We do this like this
		//because it more closely mirror how it would be done on an embedded device
		// (actually because I copied the known-good firestorm implementation)
		//		magsqr := make([]uint64, 16)
		//		magmax := uint64(0)
		//		for i := 0; i < 16; i++ {
		//			magsqr[i] = uint64(int64(qz[i])*int64(qz[i]) + int64(iz[i])*int64(iz[i]))
		//			if magsqr[i] > threshold {
		//				if magsqr[i] < magmax {
		//					break
		//				} else {
		//					magmax = magsqr[i]
		//				}
		//			}
		//		}
		txi := ra.port_to_idx[h.Primary]
		rxi := ra.port_to_idx[set]
		//		fmt.Printf("Tx: %d Rx: %d Tdx: %d Rdx: %d\n", h.Primary, set, txi, rxi)
		if isprimary == false {
			//Find the first index to be greater than half the max (quarter the square)
			//			quarter := magmax / 4
			//			less_idx := 0
			//			greater_idx := 0
			//			for i := 0; i < 16; i++ {
			//				if magsqr[i] < quarter {
			//					less_idx = i
			//				}
			//				if magsqr[i] > quarter {
			//					greater_idx = i
			//					break
			//				}
			//			}

			//			//Convert the squares into normal floating point
			//			less_val := math.Sqrt(float64(magsqr[less_idx]))
			//			greater_val := math.Sqrt(float64(magsqr[greater_idx]))
			//			half_val := math.Sqrt(float64(quarter))
			//			lerp_idx := float64(less_idx) + (half_val-less_val)/(greater_val-less_val)

			//Linearly interpolate the index (the index is related to time of flight because it is regularly sampled)

			//Fudge the result with magic_count_tx and turn into time of flight
			//			tof := (lerp_idx + float64(magic_count_tx)) / freq * 8
			tof_chip := (float64(tof_est) + magic_count_tx*256.0) / (freq * 32.0) * 1000000.0

			vz = append(vz, MSR{Mac: popHdr.Srcmac,
				Pfx:  "tof",
				Name: fmt.Sprintf("%d_to_%d", h.Primary, set),
				Time: poptime.Add(SetOffset).Add(time.Duration(SampleOffset * 8)),
				Val:  tof_chip,
				Unit: "microseconds",
			})
			_ = tof_est
			_ = intensity
			//		fmt.Printf("SEQ %d ASIC %d primary=%d\n", h.Seqno, set, h.Primary)
			//		fmt.Printf("tof: %.2f us\n", tof*1000000)
			//		fmt.Println("freq: ", freq)

			if toprint {

				//We print these just for fun / debugging, but this is not actually emitting the data
				fmt.Printf("SEQ %d ASIC %d primary=%d\n", h.Seqno, set, h.Primary)
				//				fmt.Println("lerp_idx: ", lerp_idx)
				fmt.Println("tof_sf: ", tof_sf)
				fmt.Println("freq: ", freq)
				//				fmt.Printf("tof: %.2f us\n", tof*1000000)
				fmt.Println("intensity: ", intensity)
				fmt.Println("tof chip raw: ", tof_est)
				fmt.Println("tof chip estimate: ", tof_chip)
				//				fmt.Println("tof 50us estimate: ", lerp_idx*50)
				//				fmt.Println("data: ")
				//				for i := 0; i < 16; i++ {
				//					fmt.Printf(" [%2d] %6d + %6di (%.2f)\n", i, qz[i], iz[i], math.Sqrt(float64(magsqr[i])))
				//				}
				//				fmt.Println(".")
			}
			ra.setToFs(float32(tof_chip), txi, rxi, toprint)

			ra.vel_matrix[txi][rxi] = 0.5 * (ra.s_matrix[txi][rxi]/ra.tof_matrix[txi][rxi] -
				ra.s_matrix[rxi][txi]/ra.tof_matrix[rxi][txi])

			//Append this time of flight to the output data set
			//For more "real" implementations, this would likely
			//be a rolling-window smoothed time of flight. You do not have
			//to base this value on just the data from this set and
			//you do not have to emit every time either (downsampling is ok)
			odata.Tofs = append(odata.Tofs, l7g.TOFMeasure{
				Src: int(h.Primary),
				Dst: set,
				Val: tof_chip})
		} else {
			//isprimary == true
			sum := float32(0)
			//			for i := 0; i < 16; i++ {
			//				sum = sum + float32(magsqr[i])
			//			}
			ra.trace_sum[set] = sum / float32(16.0*32768.0*32768.0)
		}

	} //end for each of the four measurements
	ra.num_samples = ra.num_samples + 1
	ra.cardinalVelocities()
	coeff := float32(0.95)
	ra.filterVelocity(coeff)
	ra.filterTrace(0.90, h.Primary)
	ra.calibrateVelocity(int32(5.0 / (1.0 - coeff)))

	if popHdr.Srcmac == "98bed0134528465a" {
		fmt.Printf("%d, %.3f, %.3f, %.3f\n", ra.num_samples, ra.vxyz_cal[0], ra.vxyz_cal[1], ra.vxyz_cal[2])
	}

	// Now we would also emit the velocities. I imagine this would use
	// the averaged/corrected time of flights that are emitted above
	// (when they are actually averaged/corrected)
	// For now, just a placeholder
	odata.Velocities = append(odata.Velocities, l7g.VelocityMeasure{X: float64(ra.vxyz_cal[0]), Y: float64(ra.vxyz_cal[1]), Z: float64(ra.vxyz_cal[2])})

	vz = append(vz, MSR{Mac: popHdr.Srcmac,
		Pfx:  "vel",
		Name: "x",
		Time: poptime.Add(30 * 2 * time.Millisecond),
		Val:  float64(ra.vxyz_cal[0]),
		Unit: "m/s",
	})
	vz = append(vz, MSR{Mac: popHdr.Srcmac,
		Pfx:  "vel",
		Name: "y",
		Time: poptime.Add(30 * 2 * time.Millisecond),
		Val:  float64(ra.vxyz_cal[1]),
		Unit: "m/s",
	})
	vz = append(vz, MSR{Mac: popHdr.Srcmac,
		Pfx:  "vel",
		Name: "z",
		Time: poptime.Add(30 * 2 * time.Millisecond),
		Val:  float64(ra.vxyz_cal[2]),
		Unit: "m/s",
	})

	// You can also add some extra data here, maybe intermittently like
	if time.Now().Sub(lastNotify) > 5*time.Second {
		odata.Extradata = append(odata.Extradata, fmt.Sprintf("anemometer %s build is %d", popHdr.Srcmac, h.Build))
		lastNotify = time.Now()
	}

	LogMSR(vz)

	//Emit the data on the SASC bus
	//emit.Data(odata)
	//	fmt.Printf(".\n")
}
