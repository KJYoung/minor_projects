export type CalTodoDay = {
    year: number;
    month: number;
    day: number | null;
};

export const MONTH_SHORT_KR = new Intl.DateTimeFormat('kr', { month: 'short' });
export const MONTH_SHORT_EN = new Intl.DateTimeFormat('en', { month: 'short' });
export const MONTH_LONG_EN = new Intl.DateTimeFormat('en', { month: 'long' });
export const A_LESS_THAN_B_CalTodoDay = (a: CalTodoDay, b: CalTodoDay) => {
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

export const GetDateTimeFormat2Django = (dt: Date, fullTime?: boolean): string => {
    if(fullTime){
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} ${dt.getHours()}:${dt.getMonth()}:${dt.getSeconds()}`
    }else{
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} 10:30:57`; // default HH:MM:SS
    }
}
export const GetDjangoDateByCalTodoDay = (dt: CalTodoDay): string => {
    if(dt.day){
        return GetDateTimeFormat2Django(new Date(dt.year, dt.month, dt.day));
    }else{
        throw Error();
    }
}

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

export const GetDateObjFromDjango = (dateString: string): Date => {
    // ex. input format: 2023-07-22T12:00:00
    return new Date(dateString);
};