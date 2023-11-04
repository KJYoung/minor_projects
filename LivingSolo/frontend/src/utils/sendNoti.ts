import {
  Store,
  NOTIFICATION_INSERTION,
  NOTIFICATION_CONTAINER,
  iNotificationDismiss,
  NOTIFICATION_TYPE,
} from 'react-notifications-component';
import { NOTI_MAX_COUNT } from './Constants';
  
export const notification: {
  insert: NOTIFICATION_INSERTION;
  container: NOTIFICATION_CONTAINER;
  dismiss: iNotificationDismiss;
} = {
  insert: 'top',
  container: 'bottom-center',
  dismiss: {
    duration: 1200,
  },
};
  
// success, danger, info, default, warning
let notiCount = 0;
const notifiactionBase = (title: string, message: string, type: NOTIFICATION_TYPE) => {
  if(notiCount >= NOTI_MAX_COUNT){
    return;
  }else{
    Store.addNotification({
      ...notification,
      title,
      message,
      type,
    });
    notiCount += 1;
    setTimeout(() => {
      notiCount -= 1;
    }, 1200);
  }
};

export const notificationSuccess = (title: string, message: string) => {
  notifiactionBase(title, message, 'success');
};

export const notificationDanger = (title: string, message: string) => {
  notifiactionBase(title, message, 'danger');
};

export const notificationDefault = (title: string, message: string) => {
  notifiactionBase(title, message, 'default');
};

export const notificationWarning = (title: string, message: string) => {
  notifiactionBase(title, message, 'warning');
};

export const notificationFailure = (title: string, message: string) => {
  notifiactionBase(title, message, 'danger');
};

export const notificationInfo = (title: string, message: string) => {
  notifiactionBase(title, message, 'info');
};
  