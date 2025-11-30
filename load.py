import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

fastf1.Cache.enable_cache('cache')
YEAR=2024
GP='Bahrain'
SESSION_TYPE='R'
OUTPUT=f"leclerc_{YEAR}_{GP.replace(' ','_')}_{SESSION_TYPE}_expanded.csv"
FETCH_TELEMETRY=True

def pick_col(df,c):
    for x in c:
        if x in df.columns:
            return x
    return None

def to_seconds(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v,pd.Timedelta):
        return v.total_seconds()
    try:
        return float(v)
    except:
        try:
            return pd.to_timedelta(v).total_seconds()
        except:
            return np.nan

session=fastf1.get_session(YEAR,GP,SESSION_TYPE)
try:
    session.load()
except:
    session.load(telemetry=False)

laps=session.laps.copy()
col_driver=pick_col(laps,['Driver','driver','DriverNumber'])
col_lapno=pick_col(laps,['LapNumber','Lap','lap_number'])
col_laptime=pick_col(laps,['LapTime','lapTime','Lap_Time','Time'])
col_s1=pick_col(laps,['Sector1Time','S1','Sector1'])
col_s2=pick_col(laps,['Sector2Time','S2','Sector2'])
col_s3=pick_col(laps,['Sector3Time','S3','Sector3'])
col_compound=pick_col(laps,['Compound','TyreCompound','compound'])
col_stint=pick_col(laps,['Stint','stint'])
col_tirelife=pick_col(laps,['TyreLife','TyreLifeLaps','TyreLifeLap'])
col_position=pick_col(laps,['Position','Pos','position'])
col_gap=pick_col(laps,['GapToLeader','Interval','Gap'])
col_session_time=pick_col(laps,['Time','Timestamp','SessionTime','LapStartTime'])
col_pitin=pick_col(laps,['PitInTime','PitIn'])
col_pitout=pick_col(laps,['PitOutTime','PitOut'])
col_pitstopdur=pick_col(laps,['PitStopTime','PitStopDuration'])

laps['driver_code']=laps[col_driver]
laps['lap_number']=laps[col_lapno]
laps['lap_time_raw']=laps[col_laptime]
laps['lap_time_s']=laps[col_laptime].apply(to_seconds) if col_laptime else np.nan
laps['s1_raw']=laps[col_s1] if col_s1 else np.nan
laps['s2_raw']=laps[col_s2] if col_s2 else np.nan
laps['s3_raw']=laps[col_s3] if col_s3 else np.nan
laps['s1_s']=laps[col_s1].apply(to_seconds) if col_s1 else np.nan
laps['s2_s']=laps[col_s2].apply(to_seconds) if col_s2 else np.nan
laps['s3_s']=laps[col_s3].apply(to_seconds) if col_s3 else np.nan
laps['compound']=laps[col_compound]
laps['stint']=laps[col_stint]
laps['tyre_life']=laps[col_tirelife] if col_tirelife else np.nan
laps['position']=laps[col_position]
laps['gap_to_leader_raw']=laps[col_gap] if col_gap else np.nan
laps['gap_to_leader_s']=laps['gap_to_leader_raw'].apply(to_seconds) if col_gap else np.nan
laps['session_time_raw']=laps[col_session_time] if col_session_time else np.nan
laps['pit_in_time']=laps[col_pitin] if col_pitin else np.nan
laps['pit_out_time']=laps[col_pitout] if col_pitout else np.nan
laps['pit_stop_dur_raw']=laps[col_pitstopdur] if col_pitstopdur else np.nan
if col_pitstopdur:
    laps['pit_stop_dur_s']=laps[col_pitstopdur].apply(to_seconds)
else:
    laps['pit_stop_dur_s']=np.nan

lec=laps[laps['driver_code']=='LEC'].copy()
lec=lec.sort_values('lap_number').reset_index(drop=True)

all_laps=session.laps.copy()
all_laps=all_laps.sort_values([col_driver if col_driver else 'Driver',col_lapno if col_lapno else 'Lap']).reset_index(drop=True)
pos_lookup={}
for _,r in all_laps.iterrows():
    d=r[col_driver] if col_driver in r.index else r.get('Driver',None)
    ln=r[col_lapno] if col_lapno in r.index else r.get('LapNumber',None)
    p=r[col_position] if col_position in r.index else r.get('Position',None)
    if pd.notna(d) and pd.notna(ln):
        pos_lookup[(str(d),int(ln))]=int(p) if pd.notna(p) else None

rows=[]
for _,lap in tqdm(lec.iterrows(),total=len(lec)):
    ln=int(lap['lap_number'])
    prev_pos=pos_lookup.get(('LEC',ln-1),None)
    curr_pos=pos_lookup.get(('LEC',ln),None)
    overtakes=(prev_pos-curr_pos) if prev_pos and curr_pos and curr_pos<prev_pos else 0

    pit_in_flag=pd.notna(lap['pit_in_time'])
    pit_out_flag=pd.notna(lap['pit_out_time'])
    pit_duration_s=lap['pit_stop_dur_s'] if not pd.isna(lap['pit_stop_dur_s']) else (to_seconds(lap['pit_out_time'])-to_seconds(lap['pit_in_time']) if (pd.notna(lap['pit_in_time']) and pd.notna(lap['pit_out_time'])) else np.nan)

    avg_speed=np.nan
    max_speed=np.nan
    std_speed=np.nan
    avg_throttle=np.nan
    avg_brake=np.nan
    avg_tyre_temp=np.nan

    if FETCH_TELEMETRY:
        try:
            lap_obj=session.laps.pick_driver('LEC').pick_lap(ln)
            tel=lap_obj.get_telemetry()
            if 'Speed' in tel.columns: speed=tel['Speed']
            elif 'speed' in tel.columns: speed=tel['speed']
            else: speed=None
            if speed is not None:
                avg_speed=float(speed.mean())
                max_speed=float(speed.max())
                std_speed=float(speed.std())
            if 'Throttle' in tel.columns: avg_throttle=float(tel['Throttle'].mean())
            elif 'throttle' in tel.columns: avg_throttle=float(tel['throttle'].mean()) if 'throttle' in tel.columns else np.nan
            if 'Brake' in tel.columns: avg_brake=float(tel['Brake'].mean())
            tyre_cols=[c for c in tel.columns if 'Tyre' in c or 'Tire' in c]
            if tyre_cols:
                avg_tyre_temp=float(tel[tyre_cols].mean().mean())
        except:
            pass

    rows.append({
        'lap_number':ln,
        'lap_time_s':lap['lap_time_s'],
        'lap_time_raw':str(lap['lap_time_raw']),
        's1_s':lap['s1_s'],
        's2_s':lap['s2_s'],
        's3_s':lap['s3_s'],
        's1_raw':str(lap['s1_raw']),
        's2_raw':str(lap['s2_raw']),
        's3_raw':str(lap['s3_raw']),
        'stint':lap['stint'],
        'compound':lap['compound'],
        'tyre_life':lap['tyre_life'],
        'position':curr_pos,
        'prev_position':prev_pos,
        'overtakes':overtakes,
        'gap_to_leader_s':lap['gap_to_leader_s'],
        'session_time_raw':str(lap['session_time_raw']),
        'pit_in':pit_in_flag,
        'pit_out':pit_out_flag,
        'pit_duration_s':pit_duration_s,
        'pit_stop_dur_s':lap['pit_stop_dur_s'],
        'avg_speed':avg_speed,
        'max_speed':max_speed,
        'std_speed':std_speed,
        'avg_throttle':avg_throttle,
        'avg_brake':avg_brake,
        'avg_tyre_temp':avg_tyre_temp
    })

df=pd.DataFrame(rows)
df=df.sort_values('lap_number').reset_index(drop=True)
df.to_csv(OUTPUT,index=False)
print("Saved",OUTPUT)
