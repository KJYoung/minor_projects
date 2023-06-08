import client from './client';

// without Payload
export const getElements = async () => {
  const response = await client.get(`/api/core/`);
  return response.data;
};

// with Payload
export const postElement = async (payload: CorePostReqType) => {
  const response = await client.post<CorePostReqType>(`/api/core/`, payload);
  return response.data;
};

// with Payload
export const putElement = async (payload: CorePutReqType) => {
  const response = await client.put<CorePutReqType>(`/api/core/${payload.id}/`, payload);
  return response.data;
};

// with Payload
export const deleteElement = async (payload: CoreDeleteReqType) => {
  const response = await client.delete<CoreDeleteReqType>(`/api/core/${payload.id}/`);
  return response.data;
};


export type CorePostReqType = {
  name: string;
};
export type CorePutReqType = {
  id: number;
  name: string;
};
export type CoreDeleteReqType = {
  id: number;
};