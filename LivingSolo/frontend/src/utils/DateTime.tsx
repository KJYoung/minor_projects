export const GetDateTimeFormat2Django = (dt: Date, fullTime?: boolean): string => {
    if(fullTime){
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} ${dt.getHours()}:${dt.getMonth()}:${dt.getSeconds()}`
    }else{
        return `${dt.getFullYear()}-${dt.getMonth()+1}-${dt.getDate()} 10:30:57`; // default HH:MM:SS
    }
}