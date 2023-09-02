import React, { useState } from "react";
import { TrxnGridHeader, TrxnGridItem } from "../../components/Trxn/TrxnGrid";
import { ViewMode } from "../TrxnMain";
import { CalMonth } from "../../utils/DateTime";
import { useSelector } from "react-redux";
import { selectTrxn } from "../../store/slices/trxn";

interface TrxnGraphicProps {
  curMonth: CalMonth,
  setCurMonth?:  React.Dispatch<React.SetStateAction<CalMonth>>,
}

export const TrxnGraphic = ({ curMonth } : TrxnGraphicProps) => {
    const [editID, setEditID] = useState(-1);
    const { elements }  = useSelector(selectTrxn);

    return <>
      <TrxnGridHeader viewMode={ViewMode.Graph}/>
      {elements && elements.map((e, index) => <TrxnGridItem key={e.id} index={index} item={e} isEditing={editID === e.id} setEditID={setEditID} viewMode={ViewMode.Graph}/>)}
    </>;
}