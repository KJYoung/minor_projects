export const DEFAULT_OPTION = '$NONE$';

// Maximum Character Count
export const TAG_NAME_LENGTH = 12;
export const TAGCLASS_NAME_LENGTH = 12;
export const TAGPRESET_NAME_LENGTH = 12;

// Maximum Notification Count
export const NOTI_MAX_COUNT = 3;

// URLS
export const TODO_URLS = (todoYear: string, todoMonth: string, todoDay: string) => `/todo?year=${todoYear}&month=${todoMonth}&day=${todoDay}`;
export const TODO_URLS_BY_DJANGO_STRING = (djangoDateString: string) => {
    // (1) Django Style string: 2023-11-03 10:30:57
    // -------------------------0123456789---------
    return TODO_URLS(djangoDateString.slice(0,4), djangoDateString.slice(5,7), djangoDateString.slice(8, 10));
};
export const TRXN_URLS = (trxnYear: string, trxnMonth: string, trxnDay: string) => `/trxn?year=${trxnYear}&month=${trxnMonth}&day=${trxnDay}`;
export const TRXN_URLS_BY_DJANGO_STRING = (djangoDateString: string) => {
    // (1) Django Style string: 2023-11-03 10:30:57
    // -------------------------0123456789---------
    return TRXN_URLS(djangoDateString.slice(0,4), djangoDateString.slice(5,7), djangoDateString.slice(8, 10));
};