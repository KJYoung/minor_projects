import React, { useState } from "react";
import { TrxnGridHeader, TrxnGridItem } from "../../components/Trxn/TrxnGrid";
import { ViewMode } from "./TrxnMain";
import { CalMonth } from "../../utils/DateTime";
import { useSelector } from "react-redux";
import { selectTrxn } from "../../store/slices/trxn";
import { styled } from "styled-components";
import { xlsxFileExporter } from "../../utils/FileExporter";

interface TrxnDetailProps {
  curMonth: CalMonth,
  setCurMonth?:  React.Dispatch<React.SetStateAction<CalMonth>>,
}

const trxnCsvDownload = () => {
  xlsxFileExporter('/api/trxn/export/', 'trxn');
};


export const TrxnDetail = ({ curMonth } : TrxnDetailProps) => {
    const [editID, setEditID] = useState(-1);
    const { elements }  = useSelector(selectTrxn);

    return <>
      <TrxnGridHeader viewMode={ViewMode.Detail}/>
      {elements && elements.map((e, index) => <TrxnGridItem key={e.id} index={index} item={e} isEditing={editID === e.id} setEditID={setEditID} viewMode={ViewMode.Detail}/>)}
      <SummaryDiv>
        <span>합계 : {elements.length !== 0 && elements.map((e) => e.amount).reduce((a, b) => a+b)}</span>
        <button onClick={trxnCsvDownload}>DOWNLOAD</button>
      </SummaryDiv>
    </>;
}

const SummaryDiv = styled.div`
  width: 100%;
  border-top: 1px solid black;
  padding: 10px;
  margin-bottom: 50px;
`;