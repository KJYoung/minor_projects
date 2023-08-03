/**
 * DateTime Utilities.
 * 
 * There are Three Types of DateTime-related Format in this Project.
 * * 1. String. Come From Django.
 * * 2. CalTodoDay. Object Created in TypeScript(Month: 0 ~ 11).
 * * 3. Date. TypeScript Native Object.
 * 
 */

export type CalTodoDay = {
    year: number;
    month: number; // Month: 0 ~ 11.
    day: number | null;
};

export const TODAY_ = new Date();
export const TOMORROW_ = new Date(TODAY_.getFullYear(), TODAY_.getMonth(), TODAY_.getDate() + 1);
export const TODAY = {year: TODAY_.getFullYear(), month: TODAY_.getMonth(), day: TODAY_.getDate()};
export const TOMORROW = {year: TOMORROW_.getFullYear(), month: TOMORROW_.getMonth(), day: TOMORROW_.getDate() + 1};

export const MONTH_SHORT_KR = new Intl.DateTimeFormat('kr', { month: 'short' });
export const MONTH_SHORT_EN = new Intl.DateTimeFormat('en', { month: 'short' });
export const MONTH_LONG_EN = new Intl.DateTimeFormat('en', { month: 'long' });

export const A_LESS_THAN_B_CalTodoDay = (a: CalTodoDay, b: CalTodoDay): boolean => {
    if(a.year < b.year){
        return true;
    }else if(a.year > b.year){
        return false;
    }else{ // a.year === b.year
        if(a.month < b.month){
            return true;
        }else if(a.month > b.month){
            return false;
        }else{ // a.month === b.month
            if(a.day !== null && b.day !== null){
                return a.day < b.day;
            }else{
                return false;
            }
        }
    }
};
export const A_EQUAL_B_CalTodoDay = (a: CalTodoDay, b: CalTodoDay): boolean => {
    return a.year === b.year && a.month === b.month && a.day === b.day;
};

export type CalMonth = {
    year: number;
    month?: number;
};

export const CUR_MONTH: CalMonth = {
    year: (new Date()).getFullYear(),
    month: (new Date()).getMonth() + 1
};
export const CUR_YEAR: CalMonth = {
    year: (new Date()).getFullYear()
};

// (3) Date => (1) Django String
export const GetDateTimeFormat2Django = (dt: Date, fullTime?: boolean): string => {
    if(fullTime){
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} ${dt.getHours()}:${dt.getMonth()}:${dt.getSeconds()}`
    }else{
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} 10:30:57`; // default HH:MM:SS
    }
};

// (2) CalTodoDay => (1) Django
export const GetDjangoDateByCalTodoDay = (dt: CalTodoDay): string => {
    if(dt.day){
        return GetDateTimeFormat2Django(new Date(dt.year, dt.month, dt.day));
    }else{
        throw Error();
    }
};

// (3) Date => (2) CalTodoDay
export const calTodoDayConst = (dt: string): CalTodoDay => {
    const regEx = new RegExp("^(\\d{4})-(\\d{2})-(\\d{2})T(\\d{2}):(\\d{2}):(\\d{2})$");
    const regGroup = regEx.exec(dt);
    if(regGroup){
        return {
            year: parseInt(regGroup[1]),
            month: parseInt(regGroup[2]) - 1,
            day: parseInt(regGroup[3]),
        };
    }else{
        throw Error;
    }
};

// (1) Django => (2) String
export const GetDateTimeFormatFromDjango = (dateString: string, compact?: boolean): string | undefined => {
    // ex. input format: 2023-07-22T12:00:00
    const regEx = new RegExp("^(\\d{4})-(\\d{2})-(\\d{2})T(\\d{2}):(\\d{2}):(\\d{2})$");
    const regGroup = regEx.exec(dateString);
    if(regGroup){
        if(compact)
            return `'${regGroup[1].slice(2)}.${regGroup[2]}.${regGroup[3]}`;
        else
            return `${regGroup[1].slice(2)}년 ${regGroup[2]}월 ${regGroup[3]}일`;
    }else{
        return `ERROR`;
    }
};

// (1) Django => (3) Date
export const GetDateObjFromDjango = (dateString: string): Date => {
    // ex. input format: 2023-07-22T12:00:00
    return new Date(dateString);
};