export const GetDateTimeFormat2Django = (dt: Date, fullTime?: boolean): string => {
    if(fullTime){
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} ${dt.getHours()}:${dt.getMonth()}:${dt.getSeconds()}`
    }else{
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} 10:30:57`; // default HH:MM:SS
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
}