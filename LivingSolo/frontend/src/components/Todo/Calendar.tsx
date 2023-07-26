import React from 'react';
import { styled } from 'styled-components';
import { CalTodoDay, MONTH_SHORT_EN, MONTH_SHORT_KR } from '../../utils/DateTime';

interface CalendarProps {
  curDay: CalTodoDay,
  setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
};

const SUNDAY_FIRST_DAYS = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];
const getDateByCalTodoDay = (calDay: CalTodoDay) => new Date(calDay.year, calDay.month, calDay.day ?? 1);
const getLastDayByCalTodoDay = (calDay: CalTodoDay) => new Date(calDay.year, calDay.month + 1, 0).getDate();
const sundayFirstMapper = (calDay: CalTodoDay) => {
    const day = new Date(calDay.year, calDay.month, calDay.day ?? 1).getDay();
    return day < 7 ? day + 1 : 0; // To Make Sunday First Calendar;
    // return day; // To Make Monday First Calendar;
};
const getDayElementLength = (calDay: CalTodoDay) => sundayFirstMapper({...calDay, day: 1}) + getLastDayByCalTodoDay(calDay);

export const Calendar = ({ curDay, setCurDay }: CalendarProps) => {
  const monthAdjuster = (calMonth: CalTodoDay, delta_month: number) => {
    if(calMonth.month + delta_month < 0){ // 0 ~ 11 Month!
        return { year: calMonth.year - 1, month: 12 + (calMonth.month + delta_month), day: null };
    }else if(calMonth.month + delta_month > 11){ // 0 ~ 11 Month!
        return { year: calMonth.year + Math.floor((calMonth.month + delta_month) / 12), month: (calMonth.month + delta_month) % 12, day: null };
    }else{
        return { year: calMonth.year, month: calMonth.month + delta_month, day: null };
    }
  };

  return <CalendarWrapper>
  <div>
    <div>
      <button onClick={() => setCurDay((cD) => monthAdjuster(cD, -1))}>
        {'<'}
      </button>
      <div>
        <span>{curDay.year}</span>
        <span>
            {MONTH_SHORT_KR.format(getDateByCalTodoDay(curDay))}
            |{MONTH_SHORT_EN.format(getDateByCalTodoDay(curDay))}
        </span>
      </div>
      <button onClick={() => setCurDay((cD) => monthAdjuster(cD, +1))}>
        {'>'}
      </button>
    </div>
    <div>
      {SUNDAY_FIRST_DAYS.map(d => 
        <span className={d} key={d}>
            {d}
        </span>
      )}
      {Array(getDayElementLength(curDay)).fill(null)
        .map((_, index) => {
            const dayOfMonth = index - sundayFirstMapper({ ...curDay, day: 1 });
            if(dayOfMonth < 0){
                return <div className='beforeFirst' key={index}></div>
            } else if(dayOfMonth < getLastDayByCalTodoDay(curDay)){
                return <div className='validDays' key={index}>
                    <span>{dayOfMonth + 1}</span>
                </div>
            } else {
                return <div className='afterLast' key={index}></div>
            }            
        })}
    </div>
  </div>
</CalendarWrapper>
};

const CalendarWrapper = styled.div`
  width: 100%;
  border: 1px solid red;
`;