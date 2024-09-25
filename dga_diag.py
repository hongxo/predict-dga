# DGA diag function was written by Pham Minh Hoang
# The function now consists of 4 methods: IEC ratio code, Rogers 4 ratios, Duval triangle #1 and pentagon #1
# Future release will include the Fuzzy Ratio and Percentage method developed by the author
# An ANN feature shall also be considered


# -------------------------------------------------------------------------------------
# IEC function
def iec(gas_content):
    H2 = gas_content["H2"]
    CH4 = gas_content["CH4"]
    C2H6 = gas_content["C2H6"]
    C2H4 = gas_content["C2H4"]
    C2H2 = gas_content["C2H2"]

    if((H2 != 0)  and  (C2H4 != 0)  and  (C2H6 != 0)):
        r1 = C2H2/C2H4
        r2 = CH4/H2
        r3 = C2H4/C2H6
    else:
        r1 = 0
        r2 = 0
        r3 = 0

    # Ratio codes
    # r1
    if(r1<0.1):
        r1_c = 0
        
    elif((r1>=0.1)  and  (r1<3)):
        r1_c = 1
        
    else:
        r1_c = 2
        
    # r2
    if(r2<0.1):
        r2_c = 1

    elif((r2>=0.1) and (r2<1)):
        r2_c = 0

    else:
        r2_c = 2

    # r3
    if(r3<1):
        r3_c = 0

    elif((r3>=1) and (r3<3)):
        r3_c = 1

    else:
        r3_c = 2

    # Rules
    # No fault
    if((r1_c == 0) and (r2_c == 0) and (r3_c == 0)):
        fault_no = 0
        fault_code = "N"

    # 1 PD low
    elif((r1_c == 0) and (r2_c == 1) and (r3_c == 0)):
        fault_no = 1
        fault_code = "PD"

    # 2 PD high
    elif((r1_c == 1) and (r2_c == 1) and (r3_c == 0)):
        fault_no = 2
        fault_code = "PD"

    # 3 D1
    elif(((r1_c == 1)or(r1_c == 2)) and (r2_c == 0) and ((r3_c == 1)or(r3_c == 2))):
        fault_no = 3
        fault_code = "D1"

    # 4 D2
    elif((r1_c == 1) and (r2_c == 0) and (r3_c == 2)):
        fault_no = 4
        fault_code = "D2"

    # 5 T < 150
    elif((r1_c == 0) and (r2_c == 0) and (r3_c == 1)):
        ault_no = 5
        fault_code = "T1"

    # 6 T = 150 - 300
    elif((r1_c == 0) and (r2_c == 2) and (r3_c == 0)):
        fault_no = 6
        fault_code = "T1"

    # 7 T = 300 - 700
    elif((r1_c == 0) and (r2_c == 2) and (r3_c == 1)):
        fault_no = 7
        fault_code = "T2"

    # //8 T > 700
    elif((r1_c == 0) and (r2_c == 2) and (r3_c == 2)):
        fault_no = 8
        fault_code = "T3"

    else:
        fault_no = 0
        fault_code = "N/A"

    return fault_code

# End of IEC function
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Rogers function
def rogers(gas_content):
    H2 = gas_content["H2"]
    CH4 = gas_content["CH4"]
    C2H6 = gas_content["C2H6"]
    C2H4 = gas_content["C2H4"]
    C2H2 = gas_content["C2H2"]

    if((H2 != 0) and (CH4 != 0) and (C2H4 != 0) and (C2H6 != 0)):
        r1 = CH4/H2
        r2 = C2H6/CH4 
        r3 = C2H4/C2H6
        r4 = C2H2/C2H4
    else:
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0

    # Ratio codes
        # r1
    if(r1<=0.1):
        r1_c = 5
            
    elif((r1>0.1)and(r1<1)):
        r1_c = 0
            
    elif((r1>=1)and(r1<3)):
        r1_c = 1
            
    else:
        r1_c = 2

    # r2 = c2h6/ch4
    if(r2<1):
        r2_c = 0

    else:
        r2_c = 1

    # r3 = c2h4/c2h6
    if(r3<1):
        r3_c = 0

    elif((r3>=1)and(r2<3)):
        r3_c = 1

    else:
        r3_c = 2

    # r4 = c2h2/c2h4
    if(r4<0.5):
        r4_c = 0

    elif((r4>=0.5)and(r4<3)):
        r4_c = 1

    else:
        r4_c = 2


    # Rules
    #0 No fault
    if((r1_c == 0)and(r2_c == 0)and(r3_c == 0)and(r4_c == 0)):
        fault_no = 0
        fault_code = "N"

    #1 PD
    elif((r1_c == 5)and(r2_c == 0)and(r3_c == 0)and(r4_c == 0)):
        fault_no = 1
        fault_code = "PD"

    # T < 150
    elif(((r1_c == 1)or(r1_c == 2))and(r2_c == 0)and(r3_c == 0)and(r4_c == 0)):
        fault_no = 2
        fault_code = "T1"

    # T = 150 - 200
    elif(((r1_c == 1)or(r1_c == 2))and(r2_c == 1)and(r3_c == 0)and(r4_c == 0)):
        fault_no = 3 
        fault_code = "T1"

    # T = 200 - 300   
    elif((r1_c == 0)and(r2_c == 1)and(r3_c == 0)and(r4_c == 0)):
        fault_no = 4
        fault_code = "T1"
        
    # General conductor overheating   
    elif((r1_c == 0)and(r2_c == 0)and(r3_c == 1)and(r4_c == 0)):
        fault_no = 5
        fault_code = "T2"

    # Circulating current   
    elif((r1_c == 1)and(r2_c == 0)and(r3_c == 1)and(r4_c == 0)):
        fault_no = 6
        fault_code = "T2"

    # Circulating current & overheated joints   
    elif((r1_c == 1)and(r2_c == 0)and(r3_c == 2)and(r4_c == 0)):
        fault_no = 7
        fault_code = "T3"

    #Flashover without power follow through
    elif((r1_c == 0)and(r2_c == 0)and(r3_c == 0)and(r4_c == 1)):
        fault_no = 8
        fault_code = "D1"

    #Arc with power follow through
    elif((r1_c == 0)and(r2_c == 0)and((r3_c == 1)or(r3_c == 2))and((r4_c == 1)or(r4_c == 2))):
        fault_no = 9
        fault_code = "D2"

    #Continous sparking to floating potential
    elif((r1_c == 0)and(r2_c == 0)and(r3_c == 2)and(r4_c == 2)):
        fault_no = 10
        fault_code = "D2"

    #PD with tracking
    elif((r1_c == 5)and(r2_c == 0)and(r3_c == 0)and((r4_c == 1)or(r4_c == 2))):
        fault_no = 11
        fault_code = "PD"

    else:
        fault_no = 0
        fault_code = "N/A"
        
    return fault_code

# End of Rogers function
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Duval triangle function
def duval_triangle(gas_content):
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    from shapely.geometry import Point, Polygon

    H2 = gas_content["H2"]
    CH4 = gas_content["CH4"]
    C2H6 = gas_content["C2H6"]
    C2H4 = gas_content["C2H4"]
    C2H2 = gas_content["C2H2"]
    
    # Avoid critical situation
    H2 = H2 + 1e-3
    CH4 = CH4 + 1e-3
    C2H6 = C2H6 + 1e-3
    C2H4 = C2H4 + 1e-3
    C2H2 = C2H2 + 1e-3

    # Create fault zones
    D1_POINT =  np.array([[0,0],
                [43.5000000000000,75.3442101292462],
                [55.0000000000000,55.4256258422041],
                [23,0]])
        
    D2_POINT =  np.array([[23,0],
                [55.0000000000000,55.4256258422041],
                [63.5000000000000,40.7031939778686],
                [55.5000000000000,26.8467875173176],
                [71,0]])                

    DT_POINT =  np.array([[71,0],
                [55.5000000000000,26.8467875173176],
                [63.5000000000000,40.7031939778686],
                [43.5000000000000,75.3442101292462],
                [48.0000000000000,83.1384387633061],
                [73,39.8371685740842],
                [67.5000000000000,30.3108891324554],
                [85,0]])

    PD_POINT =  np.array([[49.0000000000000,84.8704895708750],
                [50.0000000000000,86.6025403784439],
                [51.0000000000000,84.8704895708750]])

    T1_POINT =  np.array([[60.0000000000000,69.2820323027551],
                [58.0000000000000,65.8179306876173],
                [48.0000000000000,83.1384387633061],
                [49.0000000000000,84.8704895708750],
                [51.0000000000000,84.8704895708750]])

    T2_POINT =  np.array([[75,43.3012701892219],
                [73,39.8371685740842],
                [58.0000000000000,65.8179306876173],
                [60.0000000000000,69.2820323027551]])

    T3_POINT =  np.array([[85,0],
                [67.5000000000000,30.3108891324554],
                [75,43.3012701892219],
                [100,0]])

    # Convert points into matplotlib region
    PD_ZONE = np.transpose(PD_POINT)
    D1_ZONE = np.transpose(D1_POINT)
    D2_ZONE = np.transpose(D2_POINT)
    T1_ZONE = np.transpose(T1_POINT)
    T2_ZONE = np.transpose(T2_POINT)
    T3_ZONE = np.transpose(T3_POINT)
    DT_ZONE = np.transpose(DT_POINT)

    # Convert array into polygon by using the shapely library 
    PD_POLY = Polygon(PD_POINT)
    D1_POLY = Polygon(D1_POINT)
    D2_POLY = Polygon(D2_POINT)
    T1_POLY = Polygon(T1_POINT)
    T2_POLY = Polygon(T2_POINT)
    T3_POLY = Polygon(T3_POINT)
    DT_POLY = Polygon(DT_POINT)

    # Avoid undefined state
    if ((CH4+ C2H2 + C2H4) == 0):
        Fault_no = 0
        Fault_code = "N"
    else:
        # Create a sub-polygon from gas concentrations
        x = C2H2*100/(CH4+C2H2+C2H4) 
        y = C2H4*100/(CH4+C2H2+C2H4)
        z = CH4*100/(CH4+C2H2+C2H4)       
        
        Px = y + z*0.5
        Py = z*math.cos((math.pi)/6)
        P = np.array([Px,Py])   

        # Find the centroid of the sub-polygon
        P_CENTROID = Point(P)

        # Check where the centroid lies
        if P_CENTROID.within(PD_POLY):
            Fault_no = 1
            Fault_code = "PD"
        elif P_CENTROID.within(D1_POLY):
            Fault_no = 2
            Fault_code = "D1"
        elif P_CENTROID.within(D2_POLY):
            Fault_no = 3
            Fault_code = "D2"
        elif P_CENTROID.within(T1_POLY):
            Fault_no = 4
            Fault_code = "T1"
        elif P_CENTROID.within(T2_POLY):
            Fault_no = 5
            Fault_code = "T2"
        elif P_CENTROID.within(T3_POLY):
            Fault_no = 6
            Fault_code = "T3"
        elif P_CENTROID.within(DT_POLY):
            Fault_no = 7
            Fault_code = "DT"        
        else:
            Fault_no = 0
            Fault_code = "N/A"

        # Polygon plot
        triangle_figure, plt_triangle = plt.subplots(1,1)
        plt_triangle.scatter(x=[P_CENTROID.x],y=[P_CENTROID.y],s=50,c='r')
        plt_triangle.fill(PD_ZONE[0],PD_ZONE[1], c='k',alpha=0.5)
        plt_triangle.fill(D1_ZONE[0],D1_ZONE[1], c='g',alpha=0.5)
        plt_triangle.fill(D2_ZONE[0],D2_ZONE[1], c='m',alpha=0.5)
        plt_triangle.fill(T1_ZONE[0],T1_ZONE[1], c='0.75',alpha=0.5)
        plt_triangle.fill(T2_ZONE[0],T2_ZONE[1], c='y',alpha=0.5)
        plt_triangle.fill(T3_ZONE[0],T3_ZONE[1], c='r',alpha=0.5)
        plt_triangle.fill(DT_ZONE[0],DT_ZONE[1], c='c',alpha=0.5)
    
        # Annotation
        plt_triangle.text(52,75,'T1')
        plt_triangle.text(64,55,'T2')
        plt_triangle.text(80,20,'T3')
        plt_triangle.text(65,20,'DT')
        plt_triangle.text(20,20,'D1')
        plt_triangle.text(45,20,'D2')
        plt_triangle.axis('off')
        plt_triangle.text(50,-5,r'$C_{2}H_{2}$')
        plt_triangle.text(80,40,r'$C_{2}H_{4}$')
        plt_triangle.text(15,40,r'$CH_{4}$')

    return [Fault_code,triangle_figure]

# End of Duval triangle function
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Duval triangle function
def duval_pentagon(gas_content):
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    from shapely.geometry import Point, Polygon

    H2 = gas_content["H2"]
    CH4 = gas_content["CH4"]
    C2H6 = gas_content["C2H6"]
    C2H4 = gas_content["C2H4"]
    C2H2 = gas_content["C2H2"]

    # Create fault zones
    PD_ZONE = np.array([[-1, -1, 0, 0], [24.5, 33, 33, 24.5]])
    D1_ZONE = np.array([[0, 38, 32, 4, 0], [40, 12, -6, 16, 1.5]])
    D2_ZONE = np.array([[4, 32, 24, -1], [16, -6, -30, -2]])
    T1_ZONE = np.array([[-22.5, -6, -1, 0, -35], [-32, -4, -2, 1.5, 3]])
    T2_ZONE = np.array([[1, -6, -22.5], [-32, -4, -32]])
    T3_ZONE = np.array([[24, -1, -6, 1, 23.2], [-30, -2, -4, -32, -32]])
    S_ZONE = np.array([[-35, 0, 0, -1, -1, 0, 0, -38], [3, 1.5, 24.5, 24.5, 33, 33, 40, 12]])

    # Convert array into polygon by using the shapely library 
    PD_POLY = Polygon(np.transpose(PD_ZONE))
    D1_POLY = Polygon(np.transpose(D1_ZONE))
    D2_POLY = Polygon(np.transpose(D2_ZONE))
    T1_POLY = Polygon(np.transpose(T1_ZONE))
    T2_POLY = Polygon(np.transpose(T2_ZONE))
    T3_POLY = Polygon(np.transpose(T3_ZONE))
    S_POLY = Polygon(np.transpose(S_ZONE))

    # Avoid undefined state
    if ((CH4+ C2H2 + C2H4 + H2 + C2H6 ) == 0):
        Fault_no = 0
        Fault_code = "N"
    else:
        # Create a sub-polygon from gas concentrations
        GAS = np.array([H2, C2H6, CH4, C2H4, C2H2])
        TOTAL = sum(GAS);
        GAS_PER = GAS*100/TOTAL;
        GAS_ALPHA = np.array([90, 162, -126, -54, 18])
        x = np.array([0,0,0,0,0])
        y = np.array([0,0,0,0,0])
        for ii in range(5):
            x[ii] = GAS_PER[ii]*math.cos(math.radians(GAS_ALPHA[ii]))
            y[ii] = GAS_PER[ii]*math.cos(math.radians(90-GAS_ALPHA[ii]))
        P = np.array([x,y])   
        P_POLY = Polygon(np.transpose(P))
        
        # Find the centroid of the sub-polygon
        P_CENTROID = P_POLY.centroid

        # Check where the centroid lies
        if P_CENTROID.within(PD_POLY):
            Fault_no = 1
            Fault_code = "PD"
        elif P_CENTROID.within(D1_POLY):
            Fault_no = 2
            Fault_code = "D1"
        elif P_CENTROID.within(D2_POLY):
            Fault_no = 3
            Fault_code = "D2"
        elif P_CENTROID.within(T1_POLY):
            Fault_no = 4
            Fault_code = "T1"
        elif P_CENTROID.within(T2_POLY):
            Fault_no = 5
            Fault_code = "T2"
        elif P_CENTROID.within(T3_POLY):
            Fault_no = 6
            Fault_code = "T3"
        elif P_CENTROID.within(S_POLY):
            Fault_no = 7
            Fault_code = "S"        
        else:
            Fault_no = 0
            Fault_code = "N/A"

        # Polygon plot
        # Create fault zones
        PD_ZONE = np.array([[-1, -1, 0, 0], [24.5, 33, 33, 24.5]])
        D1_ZONE = np.array([[0, 38, 32, 4, 0], [40, 12, -6, 16, 1.5]])
        D2_ZONE = np.array([[4, 32, 24, -1], [16, -6, -30, -2]])
        T1_ZONE = np.array([[-22.5, -6, -1, 0, -35], [-32, -4, -2, 1.5, 3]])
        T2_ZONE = np.array([[1, -6, -22.5], [-32, -4, -32]])
        T3_ZONE = np.array([[24, -1, -6, 1, 23.2], [-30, -2, -4, -32, -32]])
        S_ZONE = np.array([[-35, 0, 0, -1, -1, 0, 0, -38], [3, 1.5, 24.5, 24.5, 33, 33, 40, 12]])

        # Convert array into polygon by using the shapely library 
        PD_POLY = Polygon(np.transpose(PD_ZONE))
        D1_POLY = Polygon(np.transpose(D1_ZONE))
        D2_POLY = Polygon(np.transpose(D2_ZONE))
        T1_POLY = Polygon(np.transpose(T1_ZONE))
        T2_POLY = Polygon(np.transpose(T2_ZONE))
        T3_POLY = Polygon(np.transpose(T3_ZONE))
        S_POLY = Polygon(np.transpose(S_ZONE))

        pentagon_figure, plt_pentagon = plt.subplots(1,1)
        plt_pentagon.scatter(x=[P_CENTROID.x],y=[P_CENTROID.y],s=50,c='r')
        plt_pentagon.fill(PD_ZONE[0],PD_ZONE[1], c='k',alpha=0.5)
        plt_pentagon.fill(D1_ZONE[0],D1_ZONE[1], c='g',alpha=0.5)
        plt_pentagon.fill(D2_ZONE[0],D2_ZONE[1], c='m',alpha=0.5)
        plt_pentagon.fill(T1_ZONE[0],T1_ZONE[1], c='0.75',alpha=0.5)
        plt_pentagon.fill(T2_ZONE[0],T2_ZONE[1], c='y',alpha=0.5)
        plt_pentagon.fill(T3_ZONE[0],T3_ZONE[1], c='r',alpha=0.5)
        plt_pentagon.fill(S_ZONE[0],S_ZONE[1], c='c',alpha=0.5)
        #plt_pentagon.fill(P[0],P[1], c='k',alpha=0.2)
        # Annotation
        plt_pentagon.text(-15,15,'S')
        plt_pentagon.text(-15,-5,'T1')
        plt_pentagon.text(15,15,'D1')
        plt_pentagon.text(15,-5,'D2')
        plt_pentagon.text(5,-20,'T3')
        plt_pentagon.text(-10,-20,'T2')
        plt_pentagon.axis('off')
        plt_pentagon.text(-5,42,r'$40\% H_{2}$')
        plt_pentagon.text(35,15,r'$40\% C_{2}H_{2}$')
        plt_pentagon.text(20,-35,r'$40\% C_{2}H_{4}$')
        plt_pentagon.text(-30,-35,r'$40\% CH_{4}$')
        plt_pentagon.text(-45,15,r'$40\% C_{2}H_{6}$')
        #plt_pentagon.show()
    

    return [Fault_code,pentagon_figure]

# End of Duval pentagon function
# -------------------------------------------------------------------------------------

# Key Gas function
def key_gas(gas_content):
    import numpy as np
    import matplotlib.pyplot as plt

    H2 = gas_content["H2"]
    CH4 = gas_content["CH4"]
    C2H6 = gas_content["C2H6"]
    C2H4 = gas_content["C2H4"]
    C2H2 = gas_content["C2H2"]
    CO = gas_content["CO"]
    CO2 = gas_content["CO2"]
    TCG = CO+H2+CH4+C2H6+C2H4+C2H2
    
    gas_percentage = {
        "CO": 100*CO/TCG,
        "H2": 100*H2/TCG,
        "C2H4": 100*C2H4/TCG,
        "C2H2": 100*C2H2/TCG,
        "CH4": 100*CH4/TCG,
        "C2H6": 100*C2H6/TCG
    }
    
    dominant_gas = max(gas_percentage, key=lambda item: item[1])
    if (CO != 0):
        CO_ratio = CO2/CO
        if (CO2/CO < 3) or (CO2/CO > 10):
            decomposition = 1
        else:
            decomposition = 0
    else:
        decomposition = 0    
        

    labels = ['$CO$', '$H_{2}$', '$CH_{4}$', '$C_{2}H_{6}$', '$C_{2}H_{4}$', '$C_{2}H_{2}$']
    per = [gas_percentage["CO"], gas_percentage["H2"], gas_percentage["CH4"], gas_percentage["C2H6"], gas_percentage["C2H4"], gas_percentage["C2H2"]]
    
    dominant_gas = max(gas_percentage, key=lambda item: item[1])
    if dominant_gas == "CO":
        Fault_code = "Thermal fault with paper decomposition"
    elif (dominant_gas == "C2H4" or dominant_gas == "C2H6"):
        Fault_code = "Thermal fault in oil"
    elif (dominant_gas == "H2" or dominant_gas == "CH4"):
        Fault_code = "Partial Discharge (PD)"
    elif dominant_gas == "C2H2":
        Fault_code = "Discharge"           
    else:
        Fault_code= "N/A"
    x = np.arange(len(labels))  # the label locations
    width = 0.7  # the width of the bars
    key_gas_figure, ax = plt.subplots()
    rects1 = ax.bar(x, per, width, label='Gases')
    ax.bar(x=np.arange(len(labels)),height=per,width=0.7,color=['grey', 'cyan', 'green', 'blue', 'magenta','red'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.bar_label(rects1, padding=3)
    #ax.axis('off')
    key_gas_figure.tight_layout()
    
    return [Fault_code, key_gas_figure]
# End of Key Gas function
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
