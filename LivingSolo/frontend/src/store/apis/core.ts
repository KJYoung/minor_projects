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

export type CorePostReqType = {
  id: number;
  name: string;
};