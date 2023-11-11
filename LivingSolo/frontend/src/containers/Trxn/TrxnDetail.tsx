import React, { useRef, useState } from "react";
import { TrxnGridHeader, TrxnGridItem } from "../../components/Trxn/TrxnGrid";
import { ViewMode } from "../TrxnMain";
import { CalMonth } from "../../utils/DateTime";
import { useSelector } from "react-redux";
import { selectTrxn } from "../../store/slices/trxn";
import { styled } from "styled-components";
import { workbook, xlsxFileExporter, xlsxFileImporter, xlsxUploadCallback } from "../../utils/FileInterface";
import { GeneralBtn } from "../../components/general/FuncButton";
import { TrxnExcelDialog } from "../../components/Trxn/TrxnExcelDialog";

interface TrxnDetailProps {
  curMonth: CalMonth,
  setCurMonth?:  React.Dispatch<React.SetStateAction<CalMonth>>,
};

const trxnCsvDownload = () => {
  if(window.confirm('CSV파일로 데이터를 다운로드하시겠습니까?')){
    xlsxFileExporter('/api/trxn/export/', 'trxn');
  }
};

const trxnCsvUpload = (e: React.ChangeEvent<HTMLInputElement>, xlsxUploadCallback: xlsxUploadCallback) => {
  xlsxFileImporter(e, xlsxUploadCallback);
};

export const TrxnDetail = ({ curMonth } : TrxnDetailProps) => {
    const [editID, setEditID] = useState(-1);
    const [readXlsx, setReadXlsx] = useState<workbook | null>(null);
    const [open, setOpen] = React.useState<boolean>(false);

    const handleClose = () => {
      setOpen(false);
    };
    const { elements }  = useSelector(selectTrxn);
    const invisibleFileInput = useRef(null);

    const xlsxUploadCallback = (workbook: workbook) => {
      setReadXlsx(workbook);
      setOpen(true);
    };

    return <>
      <TrxnGridHeader viewMode={ViewMode.Detail}/>
      {elements && elements.map((e, index) => <TrxnGridItem key={e.id} index={index} item={e} isEditing={editID === e.id} setEditID={setEditID} viewMode={ViewMode.Detail}/>)}
      <SummaryDiv>
        <span>합계 : {elements.length !== 0 ? elements.map((e) => e.amount).reduce((a, b) => a+b) : '0'}원</span>
        <GeneralBtn handler={trxnCsvDownload} text="Download" />
        <InvisibleFileInput ref={invisibleFileInput} type="file" onChange={(e) => trxnCsvUpload(e, xlsxUploadCallback)} accept=".xlsx" />
        <GeneralBtn handler={() => { (invisibleFileInput.current as any).click() }} text="Upload" />
      </SummaryDiv>
      {readXlsx && <TrxnExcelDialog open={open} handleClose={handleClose} workbook={readXlsx}/>}
    </>;
}

const SummaryDiv = styled.div`
  width: 100%;
  border-top: 1px solid black;
  padding: 10px;
  margin-bottom: 50px;

  display: flex;
`;

const InvisibleFileInput = styled.input`
  visibility: hidden;
`;