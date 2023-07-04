import client from './client';

// without Payload
export const getTrxns = async () => {
  const response = await client.get(`/api/trxn/`);
  return response.data;
};
export const getTrxnTypes = async () => {
  const response = await client.get(`/api/trxn/type/`);
  return response.data;
};