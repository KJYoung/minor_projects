import React from 'react';
import { styled } from 'styled-components';
import { A_LESS_THAN_B_CalTodoDay, CalTodoDay, MONTH_LONG_EN, MONTH_SHORT_KR } from '../../utils/DateTime';
import { useSelector } from 'react-redux';
import { selectTodo } from '../../store/slices/todo';
import { SatelliteManualBottom } from '../general/Satellite';


interface CalendarProps {
  curDay: CalTodoDay,
  setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
};

const TODAY_ = new Date();
const TODAY = {year: TODAY_.getFullYear(), month: TODAY_.getMonth(), day: TODAY_.getDate()};

const SUNDAY_FIRST_DAYS = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];
const getDateByCalTodoDay = (calDay: CalTodoDay) => new Date(calDay.year, calDay.month, calDay.day ?? 1);
const getLastDayByCalTodoDay = (calDay: CalTodoDay) => new Date(calDay.year, calDay.month + 1, 0).getDate();
const sundayFirstMapper = (calDay: CalTodoDay) => {
    const day = new Date(calDay.year, calDay.month, calDay.day ?? 1).getDay();
    return day < 7 ? day + 1 : 0; // To Make Sunday First Calendar;
    // return day; // To Make Monday First Calendar;
};
const getDayElementLength = (calDay: CalTodoDay) => {
    const bef_plus_valid = sundayFirstMapper({...calDay, day: 1}) + getLastDayByCalTodoDay(calDay) - 1;
    return Math.ceil(bef_plus_valid / 7) * 7;
};

export const Calendar = ({ curDay, setCurDay }: CalendarProps) => {
  const { elements } = useSelector(selectTodo);

  const monthAdjuster = (calMonth: CalTodoDay, delta_month: number) => {
    if(calMonth.month + delta_month < 0){ // 0 ~ 11 Month!
        return { year: calMonth.year - 1, month: 12 + (calMonth.month + delta_month), day: null };
    }else if(calMonth.month + delta_month > 11){ // 0 ~ 11 Month!
        return { year: calMonth.year + Math.floor((calMonth.month + delta_month) / 12), month: (calMonth.month + delta_month) % 12, day: null };
    }else{
        return { year: calMonth.year, month: calMonth.month + delta_month, day: null };
    }
  };
  const dayClickListener = (day: number) => {
    setCurDay({...curDay, day});
  };

  return <CalendarWrapper className='noselect'>
    <CalendarHeaderWrapper>
      <CalendarMonthNav className='clickable' onClick={() => setCurDay((cD) => monthAdjuster(cD, -12))}>◀︎◀︎</CalendarMonthNav>
      <CalendarMonthNav className='clickable' onClick={() => setCurDay((cD) => monthAdjuster(cD, -1))}>◀︎</CalendarMonthNav>
      <CalendarMonthWrapper>
        <CalendarMonthH1>
            <span>{curDay.year}년 {MONTH_SHORT_KR.format(getDateByCalTodoDay(curDay))}</span>
        </CalendarMonthH1>
        <CalendarMonthH2>
            <span>{MONTH_LONG_EN.format(getDateByCalTodoDay(curDay))}</span>
        </CalendarMonthH2>
      </CalendarMonthWrapper>
      <CalendarMonthNav className='clickable' onClick={() => setCurDay((cD) => monthAdjuster(cD, +1))}>▶︎</CalendarMonthNav>
      <CalendarMonthNav className='clickable' onClick={() => setCurDay((cD) => monthAdjuster(cD, +12))}>▶︎▶︎</CalendarMonthNav>
    </CalendarHeaderWrapper>

    <CalendarDatePalette>
      <CalendarDateHeader>
        {SUNDAY_FIRST_DAYS.map((d, index) => <CalendarDateElement className={`validDay ${d}`} key={index}>
            <span>{d}</span>
        </CalendarDateElement>)}
      </CalendarDateHeader>
      <CalendarDateBody>
        {Array(getDayElementLength(curDay)).fill(null)
            .map((_, index) => {
                const dayOfMonth = index - sundayFirstMapper({ ...curDay, day: 1 }) + 1;
                if(dayOfMonth < 0){ // Before the First Day.
                    return <CalendarDateElement className='beforeFirst' key={index}></CalendarDateElement>
                } else if(dayOfMonth < getLastDayByCalTodoDay(curDay)){ // Valid Days.
                    const validDay = dayOfMonth + 1;
                    const isSelected = (validDay === curDay.day) ? 'selected' : '';
                    const pastDay = A_LESS_THAN_B_CalTodoDay({...curDay, day: validDay}, TODAY) ? 'pastDay' : '';
                    const isSatSun = (index % 7 === 0) ? 'SUN' : (index % 7 === 6) ? 'SAT' : '';

                    const doneNum = elements[validDay] ? elements[validDay].filter((todo) => todo.done).length : 0;
                    const pendNum = elements[validDay] ? elements[validDay].filter((todo) => !todo.done).length : 0;
                    const totalNum = doneNum + pendNum;

                    return <CalendarDateElement className={`validDay ${isSelected} ${pastDay} ${isSatSun}`} key={index} onClick={() => dayClickListener(validDay)}>
                        <CalendarIndicatorWrapper>
                            <SatelliteManualBottom colors={elements[validDay]?.map((e) => e.color)} doneNum={doneNum} totalNum={totalNum}/>
                        </CalendarIndicatorWrapper>
                        <span>{validDay}</span>
                    </CalendarDateElement>
                } else { // After the Last Day.
                    return <CalendarDateElement className='afterLast' key={index}></CalendarDateElement>
                }            
            })}
      </CalendarDateBody>
    </CalendarDatePalette>
</CalendarWrapper>
};

const CalendarWrapper = styled.div`
  width: 800px;
  min-height: 800px;
  max-height: 800px;
  display: grid;
  grid-template-rows: 1fr 7fr;
`;

const CalendarHeaderWrapper = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 4fr 1fr 1fr;
  margin-bottom: 20px;
  border-bottom: 0.5px solid gray;
`;
const CalendarMonthNav = styled.button`
  background-color: transparent;
  border: none;
  font-size: 26px;
  color: var(--ls-gray_darker1);
`;
const CalendarMonthWrapper = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
`;
const CalendarMonthH1 = styled.div`
  font-size: 24px;
  margin-bottom: 5px;
  color: var(--ls-gray_darker1);
`;
const CalendarMonthH2 = styled.div`
  font-size: 16px;
  color: var(--ls-gray_google2);
`;

const CalendarDatePalette = styled.div`
  height: 100%;
  display: grid;
  grid-template-rows: 1fr 18fr;
  border: 0.5px solid gray;
`; 
const CalendarDateHeader = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 1fr;
`;
const CalendarDateBody = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 1fr;
`;

const CalendarDateElement = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;

  &.beforeFirst, &.afterLast {
    background-color: var(--ls-gray_lighter);
    border: 0.5px solid gray;
  };
  &.validDay {
    cursor: pointer;
    border: 0.5px solid gray;
  };
  &.pastDay {
    color: gray;
  };
  &.selected {
    background-color: khaki;
  };
  &.SUN {
    color: var(--ls-red);
  };
  &.SAT {
    color: var(--ls-blue);
  };
  &.SUN.pastDay {
    color: var(--ls-red_gray);
  };
  &.SAT.pastDay {
    color: var(--ls-blue_gray);
  };
`;

const CalendarIndicatorWrapper = styled.div`
  width: 80px;
  height: 80px;
  
  display: flex;
  justify-content: center;
  align-items: center;
`;
